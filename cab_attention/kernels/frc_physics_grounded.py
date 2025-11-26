"""
Physically-Grounded Forman-Ricci Curvature for Sparse Attention

FUNDAMENTAL PHYSICS:
====================
We want to identify edges that carry UNIQUE, HIGH-VALUE information.

PROBLEM WITH PREVIOUS APPROACH:
- FRC = A - λ*(A@A) is ADDITIVE → Low discriminative power
- At 90%+ sparsity, variance in FRC scores is too small
- Both H2O and CAB collapse because differences are negligible

NEW APPROACH: MULTIPLICATIVE UNIQUENESS
========================================
Instead of subtraction, use MULTIPLICATION to amplify differences.

Key Insight:
- Edge importance = Direct Strength × Uniqueness
- Uniqueness = 1 / (1 + Redundancy)
- This creates EXPONENTIAL discrimination, not linear

Mathematical Formula:
---------------------
FRC = A × (1 / (1 + λ × relative_redundancy))

Where:
- A[i,j]: Direct connection strength (normalized to [0,1])
- relative_redundancy[i,j] = (A@A)[i,j] / (A[i,j] + ε)
  → How many 2-hop paths exist per unit of direct connection
- λ: Controls how much we penalize redundancy

Interpretation:
- High A, low redundancy → FRC ≈ A (maximum score, KEEP!)
- High A, high redundancy → FRC < A (redundant, PRUNE unless magnitude is critical)
- Low A, low redundancy → FRC ≈ Low (weak bridge, not task-critical)
- Low A, high redundancy → FRC ≈ 0 (useless, PRUNE)

This is MULTIPLICATIVE, so differences are amplified at extreme sparsity.

PHYSICS VALIDATION:
===================
1. **Conservation**: FRC ∈ [0, A_max], bounded and stable
2. **Monotonicity**: Higher A → Higher FRC (all else equal)
3. **Redundancy Penalty**: Higher redundancy → Lower FRC (all else equal)
4. **Discriminative**: Exponential decay creates clear separation
5. **Gradient Stable**: All operations differentiable and bounded
"""

import torch
import torch.nn.functional as F


def compute_frc_multiplicative(
    A: torch.Tensor,
    lambda_redundancy: float = 1.0,
    eps: float = 1e-8,
    mode: str = 'relative'  # 'relative', 'absolute', or 'hybrid'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute FRC with multiplicative uniqueness for high discriminative power.

    Args:
        A: Affinity matrix [B, H, M, M], MUST be normalized to [0, 1]
        lambda_redundancy: Controls redundancy penalty (default 1.0)
        eps: Numerical stability constant
        mode:
            - 'relative': FRC = A × (1 / (1 + λ × (A@A/A)))  [RECOMMENDED]
            - 'absolute': FRC = A × (1 / (1 + λ × A@A))
            - 'hybrid': FRC = A × (1 - λ × tanh(A@A/A))

    Returns:
        frc_scores: [B, H, M, M], range ≈ [0, max(A)]
        redundancy: [B, H, M, M], absolute redundancy
        uniqueness: [B, H, M, M], uniqueness scores ∈ [0, 1]

    Physics:
        FRC represents "effective information flow" through edge i→j.
        High redundancy means information can flow through alternative paths,
        so the edge becomes less critical → FRC decreases multiplicatively.
    """
    # Input validation
    assert A.min() >= -eps and A.max() <= 1.0 + eps, \
        f"A must be in [0,1], got range [{A.min():.3f}, {A.max():.3f}]"

    # Compute absolute redundancy (2-hop paths)
    # redundancy[i,j] = sum_k A[i,k] * A[k,j]
    redundancy = torch.matmul(A, A)  # [B, H, M, M]

    if mode == 'relative':
        # Relative redundancy: How many 2-hop paths per unit of direct connection?
        # This is scale-invariant and highly discriminative
        relative_redundancy = redundancy / (A + eps)

        # Uniqueness: 1 / (1 + λ × relative_redundancy)
        # This is a Lorentzian/Cauchy distribution - strong penalty for high redundancy
        uniqueness = 1.0 / (1.0 + lambda_redundancy * relative_redundancy)

    elif mode == 'absolute':
        # Absolute redundancy: Simpler but less discriminative
        uniqueness = 1.0 / (1.0 + lambda_redundancy * redundancy)

    elif mode == 'hybrid':
        # Hybrid: Use tanh for bounded, smooth penalty
        # tanh maps [0, ∞) → [0, 1), providing smooth decay
        relative_redundancy = redundancy / (A + eps)
        uniqueness = 1.0 - lambda_redundancy * torch.tanh(relative_redundancy)
        uniqueness = torch.clamp(uniqueness, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # FRC = Direct Strength × Uniqueness
    # This is the KEY innovation: multiplicative, not additive
    frc_scores = A * uniqueness

    # Ensure numerical stability
    frc_scores = torch.where(
        torch.isfinite(frc_scores),
        frc_scores,
        torch.zeros_like(frc_scores)
    )

    return frc_scores, redundancy, uniqueness


def compute_frc_entropy_based(
    A: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Information-theoretic FRC using entropy.

    Intuition: An edge is important if it reduces uncertainty.
    Redundant edges don't reduce uncertainty (info already available).

    Formula:
        FRC = A × log(1 + 1/redundancy)

    This heavily penalizes high redundancy (log decay).
    """
    redundancy = torch.matmul(A, A)

    # Information gain: log(1 + 1/redundancy)
    # High redundancy → small information gain
    # Low redundancy → high information gain
    info_gain = torch.log(1.0 + 1.0 / (redundancy + eps))

    # Scale by temperature to control discrimination
    info_gain = info_gain * temperature

    # FRC = Strength × Information
    frc_scores = A * info_gain

    frc_scores = torch.where(
        torch.isfinite(frc_scores),
        frc_scores,
        torch.zeros_like(frc_scores)
    )

    return frc_scores, redundancy


def normalize_attention_matrix(
    attention: torch.Tensor,
    method: str = 'minmax',
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalize attention to [0, 1] for stable FRC computation.

    Args:
        attention: Raw attention scores [B, H, M, M] or [M, M]
        method: 'minmax', 'row', 'softmax', or 'std'
        eps: Numerical stability

    Returns:
        normalized: [B, H, M, M] or [M, M] in range [0, 1]
    """
    if method == 'minmax':
        # Global min-max normalization
        # A[i,j] = (a[i,j] - min) / (max - min)
        if attention.ndim == 4:  # [B, H, M, M]
            min_val = attention.view(attention.shape[0], attention.shape[1], -1).min(dim=-1, keepdim=True)[0]
            max_val = attention.view(attention.shape[0], attention.shape[1], -1).max(dim=-1, keepdim=True)[0]
            min_val = min_val.unsqueeze(-1)
            max_val = max_val.unsqueeze(-1)
        else:  # [M, M]
            min_val = attention.min()
            max_val = attention.max()

        normalized = (attention - min_val) / (max_val - min_val + eps)

    elif method == 'row':
        # Row normalization (probability distribution per query)
        # This preserves relative importances within each query
        row_sums = attention.sum(dim=-1, keepdim=True) + eps
        normalized = attention / row_sums

    elif method == 'softmax':
        # Softmax (most aggressive normalization)
        normalized = F.softmax(attention, dim=-1)

    elif method == 'std':
        # Standardize then sigmoid (keeps outliers)
        mean = attention.mean()
        std = attention.std() + eps
        standardized = (attention - mean) / std
        normalized = torch.sigmoid(standardized)

    else:
        raise ValueError(f"Unknown normalization: {method}")

    # Ensure [0, 1]
    normalized = torch.clamp(normalized, 0.0, 1.0)

    return normalized


def analyze_frc_discriminative_power(
    frc_scores: torch.Tensor,
    sparsity: float = 0.90
) -> dict:
    """
    Analyze if FRC scores have sufficient discriminative power for given sparsity.

    For sparse selection to work, we need:
    1. High variance in FRC scores
    2. Clear separation between kept and pruned blocks
    3. Non-degenerate distribution (not all same value)

    Returns diagnostics dict.
    """
    k_keep = max(1, int(frc_scores.numel() * (1.0 - sparsity)))

    # Get threshold
    flat_frc = frc_scores.flatten()
    threshold = torch.topk(flat_frc, k_keep, largest=True).values[-1]

    kept_scores = flat_frc[flat_frc >= threshold]
    pruned_scores = flat_frc[flat_frc < threshold]

    diagnostics = {
        'mean': flat_frc.mean().item(),
        'std': flat_frc.std().item(),
        'min': flat_frc.min().item(),
        'max': flat_frc.max().item(),
        'threshold': threshold.item(),
        'kept_mean': kept_scores.mean().item() if len(kept_scores) > 0 else 0,
        'pruned_mean': pruned_scores.mean().item() if len(pruned_scores) > 0 else 0,
        'separation': (kept_scores.mean() - pruned_scores.mean()).item() if len(kept_scores) > 0 and len(pruned_scores) > 0 else 0,
        'coefficient_of_variation': (flat_frc.std() / (flat_frc.mean() + 1e-8)).item(),
    }

    # Assess discriminative power
    # High variance and clear separation are good
    diagnostics['discriminative_power'] = diagnostics['coefficient_of_variation'] * diagnostics['separation']

    # Warnings
    if diagnostics['std'] < 0.01:
        diagnostics['warning'] = "Low variance - scores too similar!"
    elif diagnostics['separation'] < 0.01:
        diagnostics['warning'] = "Poor separation - threshold unclear!"
    elif diagnostics['coefficient_of_variation'] < 0.1:
        diagnostics['warning'] = "Low coefficient of variation - weak discrimination!"
    else:
        diagnostics['warning'] = None

    return diagnostics


def test_frc_formula_comparison():
    """
    Compare different FRC formulas to validate discriminative power.
    """
    print("="*70)
    print("FRC FORMULA COMPARISON TEST")
    print("="*70)
    print()

    # Create test affinity matrix with known structure
    M = 100
    A = torch.zeros(M, M)

    # Create 2 clusters with 1 bridge
    cluster_size = 40
    # Cluster 1: dense (high intra-cluster edges)
    A[:cluster_size, :cluster_size] = 0.8
    # Cluster 2: dense
    A[cluster_size:2*cluster_size, cluster_size:2*cluster_size] = 0.8
    # Bridge: weak connection
    bridge_idx1 = cluster_size - 1
    bridge_idx2 = cluster_size
    A[bridge_idx1, bridge_idx2] = 0.2
    A[bridge_idx2, bridge_idx1] = 0.2
    # Noise
    A += torch.randn(M, M).abs() * 0.05
    A = torch.clamp(A, 0, 1)
    A = (A + A.T) / 2  # Symmetric

    # Normalize
    A = normalize_attention_matrix(A, method='minmax')

    # Test different formulas
    print("Testing FRC formulas on synthetic 2-cluster + bridge graph...")
    print()

    # Formula 1: Additive (old, unstable)
    redundancy_old = torch.matmul(A, A) / M
    frc_additive = A - 0.5 * redundancy_old

    # Formula 2: Multiplicative relative (new)
    frc_mult_rel, _, _ = compute_frc_multiplicative(A.unsqueeze(0).unsqueeze(0), lambda_redundancy=1.0, mode='relative')
    frc_mult_rel = frc_mult_rel[0, 0]

    # Formula 3: Entropy-based
    frc_entropy, _ = compute_frc_entropy_based(A.unsqueeze(0).unsqueeze(0), temperature=1.0)
    frc_entropy = frc_entropy[0, 0]

    # Analyze each at high sparsity (95%)
    sparsity = 0.95
    print(f"Analyzing discriminative power at {sparsity*100:.0f}% sparsity:")
    print()

    print("Additive (OLD):")
    diag_add = analyze_frc_discriminative_power(frc_additive, sparsity)
    print(f"  Std: {diag_add['std']:.4f}, Separation: {diag_add['separation']:.4f}")
    print(f"  Discriminative Power: {diag_add['discriminative_power']:.4f}")
    print(f"  Warning: {diag_add['warning']}")
    print()

    print("Multiplicative Relative (NEW):")
    diag_mult = analyze_frc_discriminative_power(frc_mult_rel, sparsity)
    print(f"  Std: {diag_mult['std']:.4f}, Separation: {diag_mult['separation']:.4f}")
    print(f"  Discriminative Power: {diag_mult['discriminative_power']:.4f}")
    print(f"  Warning: {diag_mult['warning']}")
    print()

    print("Entropy-Based:")
    diag_ent = analyze_frc_discriminative_power(frc_entropy, sparsity)
    print(f"  Std: {diag_ent['std']:.4f}, Separation: {diag_ent['separation']:.4f}")
    print(f"  Discriminative Power: {diag_ent['discriminative_power']:.4f}")
    print(f"  Warning: {diag_ent['warning']}")
    print()

    # Check bridge preservation
    print("Bridge Preservation Check:")
    for name, frc in [('Additive', frc_additive),
                      ('Multiplicative', frc_mult_rel),
                      ('Entropy', frc_entropy)]:
        k_keep = max(1, int(M * M * (1 - sparsity)))
        threshold = torch.topk(frc.flatten(), k_keep, largest=True).values[-1]
        bridge_score = frc[bridge_idx1, bridge_idx2].item()
        bridge_kept = bridge_score >= threshold
        print(f"  {name:20s}: Bridge score={bridge_score:.4f}, Kept={bridge_kept}")

    print()
    print("="*70)
    print("CONCLUSION:")
    print("The formula with highest discriminative power should be used.")
    print("="*70)


if __name__ == "__main__":
    test_frc_formula_comparison()
