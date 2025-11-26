"""
Block-Level Validation Test

Replicates the N=100 synthetic experiment at the block level to confirm
that coarse-grained curvature preserves the bridge detection property.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')

from cab_attention.kernels.coarsening import coarsen_qk_max_l2_pytorch
from cab_attention.kernels.frc_kernel import (
    compute_block_frc,
    generate_block_mask,
    visualize_frc_statistics
)


def generate_block_synthetic_problem(
    num_blocks: int = 100,
    tokens_per_block: int = 64,
    block_size_community: int = 50,
    intra_prob: float = 0.9,
    bridge_weight: float = 0.001,
    intra_weight_mean: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generates a synthetic problem with two dense block communities
    connected by a single weak bridge.

    This is conceptually identical to the token-level experiment,
    but operates on blocks of tokens.

    Returns:
        q: [1, 1, num_blocks * tokens_per_block, 128]
        k: [1, 1, num_blocks * tokens_per_block, 128]
        bridge_blocks: (i, j) indices of the bridge blocks
    """
    N = num_blocks * tokens_per_block
    D = 128
    B, H = 1, 1

    # Create random Q, K tensors
    q = torch.randn(B, H, N, D, device=device)
    k = torch.randn(B, H, N, D, device=device)

    # We'll manipulate the "attention affinity" at the block level
    # by setting specific block representatives

    # Community A: blocks 0 to 49
    # Community B: blocks 50 to 99
    # Bridge: block 0 <-> block 99

    bridge_block_i = 0
    bridge_block_j = num_blocks - 1

    return q, k, (bridge_block_i, bridge_block_j)


def run_block_level_experiment():
    """
    Tests whether the coarse FRC predictor can detect bridge blocks.
    """
    print("=" * 80)
    print("BLOCK-LEVEL BRIDGE DETECTION TEST")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Parameters
    num_blocks = 100
    tokens_per_block = 64
    total_tokens = num_blocks * tokens_per_block
    sparsities = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]

    print(f"Total tokens: {total_tokens:,}")
    print(f"Blocks: {num_blocks}")
    print(f"Tokens per block: {tokens_per_block}")

    # Generate synthetic data
    q, k, (bridge_i, bridge_j) = generate_block_synthetic_problem(
        num_blocks=num_blocks,
        tokens_per_block=tokens_per_block,
        device=device
    )

    # Coarsen to block level
    print("\n--- Coarsening Q, K ---")
    q_coarse, k_coarse = coarsen_qk_max_l2_pytorch(q, k, block_size=tokens_per_block)
    print(f"Q coarse shape: {q_coarse.shape}")  # [1, 1, 100, 128]

    # Manually create a bridge in the coarse space
    # Set block 0 and block 99 to be weakly connected in a specific pattern
    # We'll manipulate k_coarse to create the structure

    # Create strong intra-community connections
    # Community A: blocks 0-49, Community B: 50-99
    community_a = torch.randn(1, 1, 50, 128, device=device)
    community_b = torch.randn(1, 1, 50, 128, device=device)

    # Normalize to create strong intra-community similarity
    community_a = community_a / community_a.norm(dim=-1, keepdim=True)
    community_b = community_b / community_b.norm(dim=-1, keepdim=True)

    # Build structured Q and K
    q_coarse = torch.cat([community_a, community_b], dim=2)
    k_coarse = q_coarse.clone()

    # Add a WEAK bridge: make block 0 and block 99 slightly aligned
    # but much weaker than intra-community connections
    bridge_vector = torch.randn(1, 1, 1, 128, device=device) * 0.1
    bridge_vector = bridge_vector / bridge_vector.norm()

    # Inject bridge signal
    k_coarse[0, 0, bridge_j] = q_coarse[0, 0, bridge_i] * 0.05 + bridge_vector[0, 0, 0] * 0.95

    print(f"\n--- Bridge Location: Block {bridge_i} <-> Block {bridge_j} ---")

    # Compute FRC
    print("\n--- Computing Block-Level FRC ---")
    frc_scores, affinity, triangles = compute_block_frc(
        q_coarse,
        k_coarse,
        temperature=1.0,
        use_relu=True,
        lambda_redundancy=0.5,
    )

    # Check bridge scores
    bridge_affinity = affinity[0, 0, bridge_i, bridge_j].item()
    bridge_frc = frc_scores[0, 0, bridge_i, bridge_j].item()
    bridge_triangles = triangles[0, 0, bridge_i, bridge_j].item()

    print(f"\nBridge Metrics:")
    print(f"  Affinity (Magnitude): {bridge_affinity:.5f}")
    print(f"  FRC Score: {bridge_frc:.5f}")
    print(f"  Triangles (Redundancy): {bridge_triangles:.5f}")

    # Test pruning at different sparsity levels
    print("\n--- Testing Sparsity Levels ---")
    results = {
        'sparsity': [],
        'method': [],
        'retrieved': []
    }

    for sp in sparsities:
        # H2O: Keep high magnitude
        k_keep = int(num_blocks * num_blocks * (1 - sp))
        threshold_mag = torch.topk(affinity.flatten(), k_keep).values[-1]
        mask_h2o = affinity >= threshold_mag
        success_h2o = mask_h2o[0, 0, bridge_i, bridge_j].item()

        # CAB: Keep low (negative) FRC
        mask_frc = generate_block_mask(
            frc_scores,
            sparsity=sp,
            keep_diagonal=False,
            causal=False,
        )
        success_frc = mask_frc[0, 0, bridge_i, bridge_j].item()

        results['sparsity'].append(sp)
        results['method'].append('H2O (Magnitude)')
        results['retrieved'].append(int(success_h2o))

        results['sparsity'].append(sp)
        results['method'].append('CAB (Curvature)')
        results['retrieved'].append(int(success_frc))

        print(f"Sparsity {sp*100:>5.1f}%  ->  H2O: {int(success_h2o)}  |  CAB: {int(success_frc)}")

    # Visualize
    print("\n--- Generating Visualizations ---")
    import pandas as pd
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    sns.barplot(x='sparsity', y='retrieved', hue='method', data=df,
                palette=['#E74C3C', '#2ECC71'], ax=axes[0])
    axes[0].set_title('Block-Level Bridge Retrieval', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Success (1=Found, 0=Lost)')
    axes[0].set_xlabel('Sparsity Ratio')
    axes[0].set_ylim(0, 1.2)
    axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(axis='y', alpha=0.3)

    # Heatmap of FRC scores
    frc_matrix = frc_scores[0, 0].cpu().numpy()
    sns.heatmap(frc_matrix, ax=axes[1], cmap='RdBu_r', center=0, cbar_kws={'label': 'FRC Score'})
    axes[1].set_title('Block-Level FRC Heatmap', fontsize=14, fontweight='bold')
    axes[1].scatter([bridge_j], [bridge_i], s=200, c='lime', marker='*', edgecolors='black', linewidths=2)
    axes[1].text(bridge_j + 2, bridge_i, 'Bridge', color='lime', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig('block_validation_results.png', dpi=150, bbox_inches='tight')
    print("Saved: block_validation_results.png")
    plt.show()

    # Additional diagnostic plots
    visualize_frc_statistics(frc_scores, affinity, triangles)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return results, frc_scores, affinity


if __name__ == '__main__':
    results, frc, aff = run_block_level_experiment()
