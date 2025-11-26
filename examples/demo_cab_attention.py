"""
End-to-End CAB-Attention Demo

Demonstrates the complete pipeline:
1. Coarsening Q, K to blocks
2. Computing FRC scores
3. Generating block masks
4. Running sparse attention
5. Visualizing geometric properties
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
sys.path.append('..')

from cab_attention import CABAttention
from cab_attention.coarse_predictor import CoarseCurvaturePredictor
from cab_attention.kernels.frc_kernel import visualize_frc_statistics


def demo_basic_usage():
    """
    Demonstrates basic CAB-Attention usage.
    """
    print("=" * 80)
    print("DEMO 1: Basic CAB-Attention Usage")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create CAB-Attention module
    attn = CABAttention(
        dim=512,
        num_heads=8,
        block_size=64,
        sparsity=0.95,
    ).to(device)

    # Generate random input
    B, N, D = 2, 4096, 512
    x = torch.randn(B, N, D, device=device)

    print(f"\nInput shape: {x.shape}")
    print(f"Config: {attn.num_heads} heads, block_size={attn.predictor.block_size}, sparsity={attn.predictor.sparsity:.1%}")

    # Forward pass with diagnostics
    print("\nRunning forward pass...")
    out, diag = attn(x, return_diagnostics=True)

    print(f"\nOutput shape: {out.shape}")
    print(f"Effective sparsity: {diag['effective_sparsity']:.2%}")
    print(f"Number of blocks (M): {diag['num_blocks']}")
    print(f"Compression ratio (N/M): {N / diag['num_blocks']:.1f}x")

    # Verify output is valid
    assert not torch.isnan(out).any(), "Output contains NaN!"
    assert out.shape == x.shape, "Output shape mismatch!"
    print("\n✓ Test passed!")


def demo_predictor_detailed():
    """
    Demonstrates the predictor in detail with visualizations.
    """
    print("\n" + "=" * 80)
    print("DEMO 2: Detailed Predictor Analysis")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create predictor
    predictor = CoarseCurvaturePredictor(
        block_size=64,
        sparsity=0.95,
        lambda_redundancy=0.5,
        use_triton=False,
    ).to(device)

    # Generate synthetic Q, K with structure
    B, H, N, D = 1, 1, 4096, 128
    print(f"\nInput: Q, K with shape [{B}, {H}, {N}, {D}]")

    # Create Q, K with block structure (simulate communities)
    num_blocks = N // 64
    q = torch.randn(B, H, N, D, device=device)
    k = torch.randn(B, H, N, D, device=device)

    # Make first half and second half form communities
    # (high intra-block similarity)
    mid = N // 2
    q[:, :, :mid] = q[:, :, :mid] / q[:, :, :mid].norm(dim=-1, keepdim=True)
    q[:, :, mid:] = q[:, :, mid:] / q[:, :, mid:].norm(dim=-1, keepdim=True)
    k[:, :, :mid] = q[:, :, :mid] + 0.1 * torch.randn_like(q[:, :, :mid])
    k[:, :, mid:] = q[:, :, mid:] + 0.1 * torch.randn_like(q[:, :, mid:])

    print("Created synthetic Q, K with two-community structure")

    # Run predictor
    print("\nRunning predictor...")
    block_mask, diagnostics = predictor(q, k, return_diagnostics=True)

    print(f"\n--- Predictor Outputs ---")
    print(f"Block mask shape: {block_mask.shape}")
    print(f"Effective sparsity: {diagnostics['effective_sparsity']:.2%}")
    print(f"Number of blocks: {diagnostics['num_blocks']}")

    # Analyze FRC statistics
    frc = diagnostics['frc_scores'][0, 0]  # [M, M]
    aff = diagnostics['affinity'][0, 0]
    tri = diagnostics['triangles'][0, 0]

    print(f"\n--- FRC Statistics ---")
    print(f"Mean FRC: {frc.mean().item():.4f}")
    print(f"Std FRC: {frc.std().item():.4f}")
    print(f"Min FRC (strongest bridge): {frc.min().item():.4f}")
    print(f"Max FRC (strongest clique): {frc.max().item():.4f}")
    print(f"% Negative curvature (bridges): {(frc < 0).float().mean().item() * 100:.1f}%")

    # Visualize
    print("\n--- Generating Visualizations ---")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Heatmaps
    # Affinity
    sns.heatmap(aff.cpu().numpy(), ax=axes[0, 0], cmap='viridis', cbar_kws={'label': 'Affinity'})
    axes[0, 0].set_title('Affinity Matrix (Direct Connections)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Key Block')
    axes[0, 0].set_ylabel('Query Block')

    # Triangles (Redundancy)
    sns.heatmap(tri.cpu().numpy(), ax=axes[0, 1], cmap='YlOrRd', cbar_kws={'label': 'Triangle Count'})
    axes[0, 1].set_title('Triangle Matrix (2-Hop Paths)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Key Block')
    axes[0, 1].set_ylabel('Query Block')

    # FRC Scores
    frc_plot = frc.cpu().numpy()
    sns.heatmap(frc_plot, ax=axes[0, 2], cmap='RdBu_r', center=0, cbar_kws={'label': 'FRC Score'})
    axes[0, 2].set_title('FRC Scores (Negative=Bridge, Positive=Clique)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Key Block')
    axes[0, 2].set_ylabel('Query Block')

    # Row 2: Distributions and scatter
    # FRC distribution
    frc_flat = frc.cpu().flatten().numpy()
    axes[1, 0].hist(frc_flat[frc_flat != 0], bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (Boundary)')
    axes[1, 0].set_xlabel('FRC Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('FRC Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')

    # Affinity vs FRC
    aff_flat = aff.cpu().flatten().numpy()
    sample_idx = np.random.choice(len(frc_flat), size=min(5000, len(frc_flat)), replace=False)
    axes[1, 1].scatter(aff_flat[sample_idx], frc_flat[sample_idx], alpha=0.3, s=5, c='purple')
    axes[1, 1].set_xlabel('Affinity (Magnitude)')
    axes[1, 1].set_ylabel('FRC Score')
    axes[1, 1].set_title('Affinity vs Curvature', fontsize=12, fontweight='bold')
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(alpha=0.3)

    # Block mask
    mask_plot = block_mask[0, 0].cpu().float().numpy()
    sns.heatmap(mask_plot, ax=axes[1, 2], cmap='Greys', cbar_kws={'label': 'Keep (1) / Prune (0)'})
    axes[1, 2].set_title(f'Block Mask (Sparsity: {diagnostics["effective_sparsity"]:.1%})', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Key Block')
    axes[1, 2].set_ylabel('Query Block')

    plt.tight_layout()
    plt.savefig('cab_attention_demo.png', dpi=150, bbox_inches='tight')
    print("Saved: cab_attention_demo.png")
    plt.show()

    print("\n✓ Visualization complete!")


def demo_flop_analysis():
    """
    Demonstrates FLOP analysis and overhead estimation.
    """
    print("\n" + "=" * 80)
    print("DEMO 3: FLOP Analysis")
    print("=" * 80)

    predictor = CoarseCurvaturePredictor(block_size=64, sparsity=0.95)

    sequence_lengths = [1024, 4096, 16384, 65536, 131072]

    print(f"\n{'N':>8} | {'M':>6} | {'Coarse':>12} | {'Matmul':>12} | {'Triangles':>12} | {'Total':>12}")
    print("-" * 80)

    for N in sequence_lengths:
        D = 128
        flop_info = predictor.estimate_flops(N, D)

        print(f"{N:>8,} | {flop_info['M']:>6} | "
              f"{flop_info['coarsening']:>12,.0f} | "
              f"{flop_info['matmul']:>12,.0f} | "
              f"{flop_info['triangles']:>12,.0f} | "
              f"{flop_info['total']:>12,.0f}")

    print("\nKey Insight: Total FLOPs scale as O(M^3) where M = N/64")
    print("For N=128k: M=2048, so M^3 ≈ 8.6B ops (trivial for modern GPUs)")


def main():
    """
    Run all demos.
    """
    print("\n" + "=" * 80)
    print("CAB-ATTENTION: END-TO-END DEMONSTRATION")
    print("=" * 80)

    # Demo 1: Basic usage
    demo_basic_usage()

    # Demo 2: Detailed predictor analysis
    demo_predictor_detailed()

    # Demo 3: FLOP analysis
    demo_flop_analysis()

    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run validation test: python cab_attention/tests/test_block_validation.py")
    print("2. Run benchmark: python benchmarks/benchmark_predictor.py")
    print("3. See README.md for full documentation")


if __name__ == '__main__':
    main()
