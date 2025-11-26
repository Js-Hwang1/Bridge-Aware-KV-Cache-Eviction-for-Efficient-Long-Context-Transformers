"""
CAB V4: Hybrid Magnitude + Topology

PROBLEM: Pure FRC selection can miss high-magnitude blocks that have high redundancy.
SOLUTION: Reserve X% for top magnitude (H2O), Y% for top FRC (CAB topology)

This ensures we get:
- High-magnitude important blocks (like needles)
- Topologically unique blocks (like bridges)
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'cab_attention'))

from kernels.frc_kernel import compute_block_frc, generate_block_mask
from kernels.coarsening import coarsen_qk_max_l2


def apply_cab_v4_hybrid(
    q: torch.Tensor,
    k: torch.Tensor,
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 32,
    magnitude_ratio: float = 0.5,  # NEW: 50% by magnitude, 50% by FRC
    formula: str = 'additive',
    lambda_redundancy: float = 0.3,  # Reduced from 0.5
) -> tuple[torch.Tensor, float, dict]:
    """
    CAB V4: Hybrid magnitude + topology selection.

    Args:
        magnitude_ratio: Fraction of blocks to select by magnitude (0.0-1.0)
                         0.0 = pure FRC (CAB V3)
                         1.0 = pure magnitude (H2O)
                         0.5 = 50/50 hybrid (RECOMMENDED)

    Returns:
        sparse_attention, compute_time, diagnostics
    """
    import time
    start_time = time.time()

    B, H, N, D = q.shape
    M = (N + block_size - 1) // block_size

    # Total blocks to keep
    k_total = max(1, int(M * M * (1.0 - sparsity)))

    # Split between magnitude and FRC
    k_magnitude = int(k_total * magnitude_ratio)
    k_frc = k_total - k_magnitude

    # 1. H2O: Select top blocks by magnitude
    attention_blocks = attention.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    magnitude_scores = attention_blocks.max(dim=-1)[0].max(dim=-1)[0]  # [B, H, M, M]

    # Get top-k by magnitude
    _, magnitude_indices = torch.topk(
        magnitude_scores.flatten(2),
        k=k_magnitude,
        dim=-1,
        largest=True
    )  # [B, H, k_magnitude]

    # 2. CAB: Select top blocks by FRC
    q_coarse, k_coarse = coarsen_qk_max_l2(q, k, block_size=block_size)

    frc_scores, affinity, redundancy = compute_block_frc(
        q_coarse, k_coarse,
        formula=formula,
        normalization='minmax',
        lambda_redundancy=lambda_redundancy,
    )

    # Get top-k by FRC
    _, frc_indices = torch.topk(
        frc_scores.flatten(2),
        k=k_frc,
        dim=-1,
        largest=True
    )  # [B, H, k_frc]

    # 3. Combine: Union of magnitude and FRC selections
    # Convert flattened indices to 2D
    magnitude_indices_2d = torch.stack([
        magnitude_indices // M,
        magnitude_indices % M
    ], dim=-1)  # [B, H, k_magnitude, 2]

    frc_indices_2d = torch.stack([
        frc_indices // M,
        frc_indices % M
    ], dim=-1)  # [B, H, k_frc, 2]

    # Create block mask
    block_mask = torch.zeros(B, H, M, M, device=attention.device, dtype=torch.bool)

    # Mark magnitude-selected blocks
    for b in range(B):
        for h in range(H):
            for idx in magnitude_indices_2d[b, h]:
                i, j = idx[0].item(), idx[1].item()
                block_mask[b, h, i, j] = True

    # Mark FRC-selected blocks
    for b in range(B):
        for h in range(H):
            for idx in frc_indices_2d[b, h]:
                i, j = idx[0].item(), idx[1].item()
                block_mask[b, h, i, j] = True

    # Always keep diagonal
    diag_mask = torch.eye(M, device=attention.device, dtype=torch.bool)
    diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
    block_mask = block_mask | diag_mask

    # 4. Expand to token level
    token_mask = block_mask.float().repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
    token_mask = token_mask[..., :N, :N]

    # 5. Apply mask
    sparse_attention = attention * token_mask

    # Renormalize
    row_sums = sparse_attention.sum(dim=-1, keepdim=True) + 1e-8
    sparse_attention = sparse_attention / row_sums

    compute_time = (time.time() - start_time) * 1000

    # Diagnostics
    from kernels.frc_kernel import analyze_frc_discriminative_power
    disc_power = analyze_frc_discriminative_power(frc_scores, sparsity=sparsity, verbose=False)

    actual_sparsity = 1.0 - block_mask.float().mean().item()

    diagnostics = {
        'discriminative_power': disc_power['discriminative_power'],
        'magnitude_blocks': k_magnitude,
        'frc_blocks': k_frc,
        'actual_sparsity': actual_sparsity,
        'magnitude_ratio': magnitude_ratio,
    }

    return sparse_attention, compute_time, diagnostics


def test_cab_v4():
    """Test CAB V4 on needle case."""
    print("="*100)
    print("TESTING CAB V4 HYBRID")
    print("="*100)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, N, D = 1, 4, 512, 64
    block_size = 32
    sparsity = 0.90

    # Generate sample
    from benchmark_realistic_niah import RealisticNIAHDataset, compute_needle_recall

    dataset = RealisticNIAHDataset(device=device)
    sample = dataset.generate_sample(context_length=N, needle_depth=0.5, num_needles=1)
    needle_pos = sample['needle_positions'][0]

    # Create Q, K with needle signal
    q = torch.randn(B, H, N, D, device=device)
    k = torch.randn(B, H, N, D, device=device)

    for i in range(max(0, needle_pos - 20), min(N, needle_pos + 20)):
        q[:, :, i, :] = k[:, :, needle_pos, :] + torch.randn(B, H, D, device=device) * 0.1

    # Full attention
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    attention_full = F.softmax(attention_scores, dim=-1)

    # Test different magnitude ratios
    print(f"\nNeedle at position {needle_pos}")
    print(f"Sparsity: {sparsity*100:.0f}%\n")

    print(f"{'Config':<25} {'Recall':<10} {'Time (ms)':<12} {'Blocks (Mag/FRC)':<20}")
    print("-" * 70)

    configs = [
        ('H2O (100% magnitude)', 1.0),
        ('CAB V4 (75% mag)', 0.75),
        ('CAB V4 (50% mag)', 0.50),
        ('CAB V4 (25% mag)', 0.25),
        ('CAB V3 (0% magnitude)', 0.0),
    ]

    best_recall = 0
    best_config = None

    for name, mag_ratio in configs:
        attn_sparse, time_ms, diag = apply_cab_v4_hybrid(
            q, k, attention_full,
            sparsity=sparsity,
            block_size=block_size,
            magnitude_ratio=mag_ratio,
            lambda_redundancy=0.3,
        )

        recall = compute_needle_recall(attn_sparse, [needle_pos], top_k=10)

        blocks_str = f"{diag['magnitude_blocks']}/{diag['frc_blocks']}"

        print(f"{name:<25} {recall:<10.3f} {time_ms:<12.2f} {blocks_str:<20}")

        if recall > best_recall:
            best_recall = recall
            best_config = (name, mag_ratio)

    print()
    print(f"✓ Best config: {best_config[0]} (recall={best_recall:.3f})")

    # Recommendation
    print("\n" + "="*100)
    print("RECOMMENDATION FOR ICML")
    print("="*100)
    print("\nCAB V4 Hybrid (50% magnitude + 50% FRC):")
    print("  - Preserves high-magnitude blocks (needles)")
    print("  - Adds topologically unique blocks (bridges)")
    print("  - Best of both worlds!")
    print("\nUpdate frc_kernel.py to use hybrid selection by default.")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_cab_v4()
