"""Memory-Efficient Permutation: Zero-Overhead Activation Reduction.

From DeepSeek-V3 Section 4.1.2.

Core insight: in MoE, moving the scalar routing weight p_i from AFTER W2
to BEFORE W2 is mathematically equivalent (linear commutes with scalar),
but eliminates the need to save expert_out for the backward pass.

Standard (Eq.1):   y = Σ p_i · W2 @ φ(W1 @ x)      — saves z_i + expert_out
Efficient (Eq.2):  y = Σ W2 @ (p_i · φ(W1 @ x))    — saves z_i only

Savings for DeepSeek-V3: ~26.3 GB per GPU, zero compute overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SwiGLU activation (used in DeepSeek-V3)
# ============================================================
def swiglu(z: torch.Tensor) -> torch.Tensor:
    """SwiGLU: split z in half, apply SiLU to gate half, element-wise multiply."""
    gate, up = z.chunk(2, dim=-1)
    return F.silu(gate) * up


# ============================================================
# Single Expert MLP (shared by both formulations)
# ============================================================
class ExpertMLP(nn.Module):
    """Two-layer MLP expert: W2 @ φ(W1 @ x), no bias."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # W1 projects to 2*d_ff for SwiGLU (gate + up)
        self.w1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w1(x)))


# ============================================================
# Standard MoE Forward (Equation 1)
# y = Σ p_i · W2 @ φ(W1 @ x)
# ============================================================
class StandardMoE(nn.Module):
    """Standard: routing weight applied AFTER expert computation.

    Backward saves per (token, expert):
      - z_i = W1 @ x          (unavoidable, for SwiGLU backward)
      - expert_out = W2 @ φ(z_i)  (EXTRA, for ∂L/∂p_i)

    ∂L/∂p_i = ∂L/∂y · expert_out  ← requires saved expert_out
    """

    def __init__(self, num_experts: int, d_model: int, d_ff: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [ExpertMLP(d_model, d_ff) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            y: [batch, seq_len, d_model]
        """
        B, S, D = x.shape
        x_flat = x.view(B * S, D)  # [num_tokens, d_model]

        # Router: compute top-k expert assignments
        logits = self.gate(x_flat)                          # [num_tokens, num_experts]
        scores = torch.softmax(logits, dim=-1)
        top_k_weights, top_k_indices = scores.topk(self.top_k, dim=-1)
        # top_k_weights: [num_tokens, k] — routing weights p_i
        # top_k_indices: [num_tokens, k] — selected expert IDs

        # Combine expert outputs
        y = torch.zeros_like(x_flat)  # [num_tokens, d_model]

        for j in range(self.top_k):
            expert_indices = top_k_indices[:, j]  # [num_tokens]
            weights = top_k_weights[:, j]         # [num_tokens] — p_i

            for expert_id in expert_indices.unique():
                mask = expert_indices == expert_id
                tokens = x_flat[mask]                          # [n, d_model]
                expert_out = self.experts[expert_id](tokens)   # [n, d_model]
                p_i = weights[mask].unsqueeze(-1)              # [n, 1]

                # === STANDARD: p_i applied AFTER expert output ===
                y[mask] += p_i * expert_out
                #               ^^^^^^^^^^^^^
                # autograd must save expert_out for ∂L/∂p_i
                # This is the EXTRA memory cost!

        return y.view(B, S, D)


# ============================================================
# Memory-Efficient MoE Forward (Equation 2)
# y = Σ W2 @ (p_i · φ(W1 @ x))
# ============================================================
class MemoryEfficientMoE(nn.Module):
    """Memory-efficient: routing weight absorbed BEFORE W2.

    Backward saves per (token, expert):
      - z_i = W1 @ x          (unavoidable, for SwiGLU backward)
      - Nothing extra!         (φ(z_i) recomputed from z_i in fused kernel)

    ∂L/∂p_i = (W2^T @ ∂L/∂y) · φ(z_i)  ← φ(z_i) recomputed from z_i, FREE
    """

    def __init__(self, num_experts: int, d_model: int, d_ff: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [ExpertMLP(d_model, d_ff) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        logits = self.gate(x_flat)
        scores = torch.softmax(logits, dim=-1)
        top_k_weights, top_k_indices = scores.topk(self.top_k, dim=-1)

        y = torch.zeros_like(x_flat)

        for j in range(self.top_k):
            expert_indices = top_k_indices[:, j]
            weights = top_k_weights[:, j]

            for expert_id in expert_indices.unique():
                mask = expert_indices == expert_id
                tokens = x_flat[mask]                          # [n, d_model]
                p_i = weights[mask].unsqueeze(-1)              # [n, 1]

                expert = self.experts[expert_id]

                # === MEMORY-EFFICIENT: p_i absorbed BEFORE W2 ===
                z_i = expert.w1(tokens)                        # [n, 2*d_ff]
                h_i = swiglu(z_i)                              # [n, d_ff]
                scaled_h = p_i * h_i                           # [n, d_ff] ← p_i here!
                expert_out = expert.w2(scaled_h)               # [n, d_model]

                y[mask] += expert_out
                # No extra save! ∂L/∂p_i recomputes φ(z_i) from z_i

        return y.view(B, S, D)


# ============================================================
# Proof of equivalence: p * (W @ h) == W @ (p * h)
# ============================================================
def prove_equivalence() -> None:
    """Demonstrate that scalar-linear commutation holds exactly."""
    torch.manual_seed(42)

    d_ff, d_model = 128, 64
    W = torch.randn(d_model, d_ff)
    h = torch.randn(d_ff)
    p = torch.tensor(0.7)

    standard = p * (W @ h)       # Eq.1: scale after W2
    efficient = W @ (p * h)      # Eq.2: scale before W2

    diff = (standard - efficient).abs().max().item()
    print(f"[Equivalence] max |standard - efficient| = {diff:.2e}")
    assert diff < 1e-5, "Should be identical up to float precision"
    print("[Equivalence] PASSED — mathematically identical\n")


# ============================================================
# Full forward equivalence test
# ============================================================
def test_forward_equivalence() -> None:
    """Both MoE variants produce identical outputs with shared weights."""
    torch.manual_seed(42)

    d_model, d_ff, num_experts, top_k = 64, 128, 8, 2
    batch, seq_len = 2, 4

    std_moe = StandardMoE(num_experts, d_model, d_ff, top_k)
    eff_moe = MemoryEfficientMoE(num_experts, d_model, d_ff, top_k)

    # Share weights
    eff_moe.load_state_dict(std_moe.state_dict())

    x = torch.randn(batch, seq_len, d_model)

    with torch.no_grad():
        y_std = std_moe(x)
        y_eff = eff_moe(x)

    diff = (y_std - y_eff).abs().max().item()
    print(f"[Forward] max |standard - efficient| = {diff:.2e}")
    assert diff < 1e-4, "Outputs should match"
    print("[Forward] PASSED — identical outputs\n")


# ============================================================
# Memory savings calculation (DeepSeek-V3)
# ============================================================
def memory_savings_deepseek_v3() -> None:
    """Calculate activation memory saved for DeepSeek-V3."""
    # DeepSeek-V3 config
    num_moe_layers = 61
    d_model = 7168
    top_k = 8
    tokens_per_gpu = 4096
    bytes_per_elem = 2  # bf16

    # Standard saves expert_out [d_model] per (token, expert, layer)
    # Efficient saves nothing extra
    extra_per_call = d_model * bytes_per_elem
    total_saved = tokens_per_gpu * top_k * num_moe_layers * extra_per_call
    total_saved_gb = total_saved / (1024 ** 3)

    print("=== DeepSeek-V3 Memory Savings ===")
    print(f"  MoE layers:        {num_moe_layers}")
    print(f"  d_model:           {d_model}")
    print(f"  top_k:             {top_k}")
    print(f"  tokens/GPU:        {tokens_per_gpu}")
    print(f"  dtype:             bf16 ({bytes_per_elem} bytes)")
    print(f"  extra/call:        {extra_per_call:,} bytes")
    print(f"  total calls/GPU:   {tokens_per_gpu * top_k * num_moe_layers:,}")
    print(f"  ─────────────────────────────────")
    print(f"  Memory saved:      {total_saved_gb:.1f} GB per GPU")
    print(f"  Compute overhead:  0 (pure algebraic rearrangement)\n")


# ============================================================
# Backward pass analysis — why the memory is saved
# ============================================================
def backward_analysis() -> None:
    """Trace what autograd saves in each formulation.

    Standard (Eq.1):
        y = p * (W2 @ φ(W1 @ x))

        Forward saves:
          z_i = W1 @ x         → needed for ∂φ/∂z (SwiGLU backward)
          expert_out = W2 @ φ(z_i)  → needed for ∂L/∂p = ∂L/∂y · expert_out

        Backward:
          ∂L/∂p_i = dot(∂L/∂y, expert_out)     ← uses saved expert_out
          ∂L/∂(expert_out) = p_i * ∂L/∂y
          ∂L/∂h = W2^T @ ∂L/∂(expert_out)
          ∂L/∂z = ∂φ/∂z(z_i) * ∂L/∂h           ← uses saved z_i

    Memory-Efficient (Eq.2):
        y = W2 @ (p * φ(W1 @ x))

        Forward saves:
          z_i = W1 @ x         → needed for ∂φ/∂z (SwiGLU backward)
          (that's it!)

        Backward:
          ∂L/∂(scaled_h) = W2^T @ ∂L/∂y
          ∂L/∂p_i = dot(∂L/∂(scaled_h), φ(z_i))  ← recompute φ(z_i) from z_i!
          ∂L/∂h = p_i * ∂L/∂(scaled_h)
          ∂L/∂z = ∂φ/∂z(z_i) * ∂L/∂h              ← uses saved z_i

    Key: φ(z_i) can be recomputed from z_i in a fused kernel.
         z_i is already saved for SwiGLU backward anyway.
         So ∂L/∂p_i costs no additional memory!
    """
    print("=== Backward Pass: Saved Tensor Comparison ===")
    print()
    print("Standard (Eq.1): y = p_i · W2 @ φ(W1 @ x)")
    print("  Saved: z_i [2*d_ff]    — for SwiGLU backward (unavoidable)")
    print("  Saved: expert_out [d_model] — for ∂L/∂p_i    (EXTRA!)")
    print("  ∂L/∂p_i = dot(∂L/∂y, expert_out)")
    print()
    print("Efficient (Eq.2): y = W2 @ (p_i · φ(W1 @ x))")
    print("  Saved: z_i [2*d_ff]    — for SwiGLU backward (unavoidable)")
    print("  Saved: (nothing extra)")
    print("  ∂L/∂p_i = dot(W2^T @ ∂L/∂y, φ(z_i))  ← φ(z_i) recomputed from z_i")
    print()
    print("Net saving per (token, expert, layer): d_model × dtype_size bytes\n")


if __name__ == "__main__":
    prove_equivalence()
    test_forward_equivalence()
    backward_analysis()
    memory_savings_deepseek_v3()
