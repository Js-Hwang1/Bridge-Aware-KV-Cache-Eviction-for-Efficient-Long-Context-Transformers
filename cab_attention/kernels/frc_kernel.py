"""
Task 1.2: Forman-Ricci Curvature Computation on Coarse Block Graph

Computes FRC on the M x M coarse adjacency matrix where M << N.
This is the key innovation: geometric selection at block level is O(M^2) not O(N^2).
"""

import torch
import torch.nn.functional as F


def compute_block_frc(
    q_coarse: torch.Tensor,
    k_coarse: torch.Tensor,
    temperature: float = 1.0,
    use_relu: bool = True,
    lambda_redundancy: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Forman-Ricci Curvature on the coarse block graph.

    Algorithm:
    1. Compute coarse affinity matrix: A = (Q_coarse @ K_coarse^T) / sqrt(d)
    2. Apply ReLU or softmax to get "potential flow"
    3. Compute node strengths (degrees)
    4. Compute triangle counts: T = A @ A (paths of length 2)
    5. Apply FRC formula: F = 4*A - S_i - S_j + 3*T
    6. Generate curvature scores (lower = more negative = bridge = KEEP)

    Args:
        q_coarse: Coarse query embeddings [B, H, M, D]
        k_coarse: Coarse key embeddings [B, H, M, D]
        temperature: Scaling factor for attention (default: 1/sqrt(D))
        use_relu: If True, use ReLU; else use softmax for affinity
        lambda_redundancy: Weight for triangle penalty term

    Returns:
        frc_scores: Forman-Ricci curvature scores [B, H, M, M]
        affinity: The coarse affinity matrix [B, H, M, M]
        triangles: Triangle counts [B, H, M, M]

    Example:
        >>> q_coarse = torch.randn(2, 8, 2000, 128, device='cuda')
        >>> k_coarse = torch.randn(2, 8, 2000, 128, device='cuda')
        >>> frc, aff, tri = compute_block_frc(q_coarse, k_coarse)
        >>> frc.shape  # (2, 8, 2000, 2000)
    """
    B, H, M, D = q_coarse.shape
    assert k_coarse.shape == (B, H, M, D)

    # 1. Compute Coarse Affinity Matrix
    # A = Q @ K^T / sqrt(D)  [B, H, M, M]
    scale = temperature / (D ** 0.5)
    affinity = torch.matmul(q_coarse, k_coarse.transpose(-2, -1)) * scale

    # 2. Apply activation to enforce sparsity/positivity
    if use_relu:
        # ReLU: Cheap, enforces sparsity, avoids expensive exp
        A = F.relu(affinity)
    else:
        # Softmax: More standard for attention, but more expensive
        A = F.softmax(affinity, dim=-1)

    # 3. Compute Node Strengths (Weighted Degree)
    # S_i = sum over j of A[i, j]  [B, H, M]
    node_strength = torch.sum(A, dim=-1)  # [B, H, M]

    # Broadcast to edge dimensions
    S_i = node_strength.unsqueeze(-1).expand(-1, -1, -1, M)  # [B, H, M, M]
    S_j = node_strength.unsqueeze(-2).expand(-1, -1, M, -1)  # [B, H, M, M]

    # 4. Compute Triangles (Redundancy Term)
    # T[i,j] = sum_k A[i,k] * A[k,j]
    # This counts 2-hop paths from i to j (shared neighbors)
    triangles = torch.matmul(A, A)  # [B, H, M, M]

    # 5. Forman-Ricci Curvature Formula
    # Standard FRC: F = 4*w_ij - S_i - S_j + (triangle support)
    # We use a simplified version optimized for attention:
    #   FRC = Direct - Redundancy
    #   where Direct = A[i,j]
    #   and Redundancy = lambda * Triangles[i,j]
    #
    # More negative FRC = stronger bridge = KEEP
    # More positive FRC = redundant clique = PRUNE

    # Augmented Forman formula
    frc_scores = 4.0 * A - S_i - S_j + 3.0 * triangles

    # For selection: we want to KEEP edges with LOWEST (most negative) scores
    # So we'll use -FRC as the importance metric in downstream selection

    return frc_scores, affinity, triangles


def generate_block_mask_from_frc(
    frc_scores: torch.Tensor,
    sparsity: float = 0.95,
    keep_diagonal: bool = True,
    causal: bool = False,
) -> torch.Tensor:
    """
    Generates a binary block mask from FRC scores.

    Strategy:
    - For each query block i, select top-k key blocks j based on LOWEST FRC
      (most negative = strongest bridge = most important)
    - Optionally enforce causal masking
    - Optionally always keep diagonal blocks (local attention)

    Args:
        frc_scores: FRC scores [B, H, M, M]
        sparsity: Fraction of blocks to PRUNE (0.95 = keep only 5%)
        keep_diagonal: Always include diagonal blocks (local attention)
        causal: Enforce causal masking (no future tokens)

    Returns:
        block_mask: Binary mask [B, H, M, M] where 1 = KEEP, 0 = PRUNE

    Example:
        >>> frc = torch.randn(2, 8, 2000, 2000, device='cuda')
        >>> mask = generate_block_mask_from_frc(frc, sparsity=0.95)
        >>> mask.float().mean()  # Should be ~0.05 (5% of blocks kept)
    """
    B, H, M, _ = frc_scores.shape

    # Calculate number of blocks to KEEP per query
    k = max(1, int(M * (1.0 - sparsity)))

    # For each query block, find top-k KEY blocks with LOWEST FRC
    # (most negative curvature = bridges)
    # We use topk with largest=False to get the smallest values
    _, top_indices = torch.topk(frc_scores, k, dim=-1, largest=False, sorted=False)

    # Create mask
    mask = torch.zeros_like(frc_scores, dtype=torch.bool)

    # Scatter 1s at the selected indices
    # We need to expand indices for scatter
    batch_idx = torch.arange(B, device=frc_scores.device).view(B, 1, 1, 1).expand(B, H, M, k)
    head_idx = torch.arange(H, device=frc_scores.device).view(1, H, 1, 1).expand(B, H, M, k)
    query_idx = torch.arange(M, device=frc_scores.device).view(1, 1, M, 1).expand(B, H, M, k)

    mask[batch_idx, head_idx, query_idx, top_indices] = True

    # Enforce diagonal (local attention)
    if keep_diagonal:
        diag_mask = torch.eye(M, device=frc_scores.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        mask = mask | diag_mask

    # Enforce causal masking if requested
    if causal:
        causal_mask = torch.tril(torch.ones(M, M, device=frc_scores.device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        mask = mask & causal_mask

    return mask


def visualize_frc_statistics(frc_scores: torch.Tensor, affinity: torch.Tensor, triangles: torch.Tensor):
    """
    Helper function to analyze FRC distribution.
    Useful for debugging and understanding the geometric properties.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Move to CPU and flatten
    frc_flat = frc_scores.detach().cpu().flatten().numpy()
    aff_flat = affinity.detach().cpu().flatten().numpy()
    tri_flat = triangles.detach().cpu().flatten().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # FRC distribution
    axes[0].hist(frc_flat[frc_flat != 0], bins=100, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (Bridge/Clique Boundary)')
    axes[0].set_xlabel('FRC Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Forman-Ricci Curvature Distribution')
    axes[0].legend()
    axes[0].set_yscale('log')

    # Affinity vs FRC scatter
    sample_idx = np.random.choice(len(frc_flat), size=min(10000, len(frc_flat)), replace=False)
    axes[1].scatter(aff_flat[sample_idx], frc_flat[sample_idx], alpha=0.3, s=1)
    axes[1].set_xlabel('Affinity (Direct Connection)')
    axes[1].set_ylabel('FRC Score')
    axes[1].set_title('Affinity vs Curvature')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)

    # Triangles vs FRC
    axes[2].scatter(tri_flat[sample_idx], frc_flat[sample_idx], alpha=0.3, s=1, color='green')
    axes[2].set_xlabel('Triangle Count (Redundancy)')
    axes[2].set_ylabel('FRC Score')
    axes[2].set_title('Redundancy vs Curvature')
    axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\n=== FRC Statistics ===")
    print(f"Mean FRC: {np.mean(frc_flat):.4f}")
    print(f"Std FRC: {np.std(frc_flat):.4f}")
    print(f"Min FRC (Most Negative/Bridge): {np.min(frc_flat):.4f}")
    print(f"Max FRC (Most Positive/Clique): {np.max(frc_flat):.4f}")
    print(f"% Negative (Bridges): {100 * np.mean(frc_flat < 0):.2f}%")
    print(f"% Positive (Cliques): {100 * np.mean(frc_flat > 0):.2f}%")
