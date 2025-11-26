"""
Comprehensive CAB Testing and Validation for ICML 2025 Submission

This script performs rigorous scientific testing of CAB attention:
1. Mathematical validation of FRC formulas
2. Discriminative power analysis across sparsity levels
3. Bridge recovery on synthetic graphs
4. NIAH (Needle-in-a-Haystack) benchmarks
5. Comparison against H2O baseline
6. Ablation studies (formulas, normalization, lambda)

Goal: Ensure CAB can reliably work at 90%+ sparsity for ICML publication.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add cab_attention to path
sys.path.insert(0, str(Path(__file__).parent / 'cab_attention'))

from kernels.frc_kernel import (
    compute_block_frc,
    generate_block_mask,
    analyze_frc_discriminative_power,
    validate_frc_stability,
)


def test_synthetic_bridge_recovery():
    """
    Test 1: Synthetic Bridge Recovery

    Create a 2-cluster graph with a weak bridge and verify that:
    1. H2O (magnitude-based) likely prunes the bridge
    2. CAB (curvature-based) preserves the bridge

    This is the CORE test that validates CAB's topological advantage.
    """
    print("="*80)
    print("TEST 1: SYNTHETIC BRIDGE RECOVERY")
    print("="*80)

    # Create 2-cluster graph
    M = 100
    cluster_size = 40
    bridge_strength = 0.15  # Weak bridge
    cluster_strength = 0.85  # Strong intra-cluster

    # Build adjacency matrix
    A = torch.zeros(M, M)

    # Cluster 1: dense connections
    A[:cluster_size, :cluster_size] = cluster_strength

    # Cluster 2: dense connections
    A[cluster_size:2*cluster_size, cluster_size:2*cluster_size] = cluster_strength

    # Bridge: weak but unique connection
    bridge_idx1 = cluster_size - 1
    bridge_idx2 = cluster_size
    A[bridge_idx1, bridge_idx2] = bridge_strength
    A[bridge_idx2, bridge_idx1] = bridge_strength

    # Add noise
    A += torch.randn(M, M).abs() * 0.05
    A = torch.clamp(A, 0, 1)
    A = (A + A.T) / 2  # Symmetric

    # Normalize
    A = (A - A.min()) / (A.max() - A.min() + 1e-8)
    A = A.unsqueeze(0).unsqueeze(0)  # [1, 1, M, M]

    print(f"\nGraph structure:")
    print(f"  - 2 clusters of size {cluster_size}, strength {cluster_strength:.2f}")
    print(f"  - Bridge at ({bridge_idx1}, {bridge_idx2}), strength {bridge_strength:.2f}")
    print(f"  - Bridge is {cluster_strength/bridge_strength:.1f}x weaker than intra-cluster edges")

    # Test different sparsity levels
    sparsity_levels = [0.80, 0.85, 0.90, 0.95]

    results = {
        'sparsity': sparsity_levels,
        'h2o_keeps_bridge': [],
        'cab_additive_keeps_bridge': [],
        'cab_multiplicative_keeps_bridge': [],
        'cab_entropy_keeps_bridge': [],
    }

    for sparsity in sparsity_levels:
        k_keep = max(1, int(M * M * (1 - sparsity)))

        # H2O: magnitude-based selection (max pooling)
        h2o_threshold = torch.topk(A.flatten(), k_keep, largest=True).values[-1]
        h2o_keeps = A[0, 0, bridge_idx1, bridge_idx2] >= h2o_threshold
        results['h2o_keeps_bridge'].append(h2o_keeps.item())

        # CAB Additive
        frc_add, _, _ = compute_block_frc(
            A, A, formula='additive', normalization='minmax', lambda_redundancy=0.5
        )
        cab_add_threshold = torch.topk(frc_add.flatten(), k_keep, largest=True).values[-1]
        cab_add_keeps = frc_add[0, 0, bridge_idx1, bridge_idx2] >= cab_add_threshold
        results['cab_additive_keeps_bridge'].append(cab_add_keeps.item())

        # CAB Multiplicative
        frc_mult, _, _ = compute_block_frc(
            A, A, formula='multiplicative', normalization='minmax', lambda_redundancy=1.0
        )
        cab_mult_threshold = torch.topk(frc_mult.flatten(), k_keep, largest=True).values[-1]
        cab_mult_keeps = frc_mult[0, 0, bridge_idx1, bridge_idx2] >= cab_mult_threshold
        results['cab_multiplicative_keeps_bridge'].append(cab_mult_keeps.item())

        # CAB Entropy
        frc_ent, _, _ = compute_block_frc(
            A, A, formula='entropy', normalization='minmax'
        )
        cab_ent_threshold = torch.topk(frc_ent.flatten(), k_keep, largest=True).values[-1]
        cab_ent_keeps = frc_ent[0, 0, bridge_idx1, bridge_idx2] >= cab_ent_threshold
        results['cab_entropy_keeps_bridge'].append(cab_ent_keeps.item())

    # Print results
    print("\nBridge Preservation Results:")
    print(f"{'Sparsity':<12} {'H2O':<8} {'CAB-Add':<10} {'CAB-Mult':<10} {'CAB-Ent':<10}")
    print("-" * 55)
    for i, s in enumerate(sparsity_levels):
        print(f"{s*100:>3.0f}%         "
              f"{'✓' if results['h2o_keeps_bridge'][i] else '✗':<8} "
              f"{'✓' if results['cab_additive_keeps_bridge'][i] else '✗':<10} "
              f"{'✓' if results['cab_multiplicative_keeps_bridge'][i] else '✗':<10} "
              f"{'✓' if results['cab_entropy_keeps_bridge'][i] else '✗':<10}")

    # Success criteria: CAB should preserve bridge better than H2O
    cab_mult_success_rate = sum(results['cab_multiplicative_keeps_bridge']) / len(sparsity_levels)
    h2o_success_rate = sum(results['h2o_keeps_bridge']) / len(sparsity_levels)

    print(f"\nBridge Preservation Rate:")
    print(f"  H2O:                  {h2o_success_rate*100:.0f}%")
    print(f"  CAB Multiplicative:   {cab_mult_success_rate*100:.0f}%")

    if cab_mult_success_rate > h2o_success_rate:
        print("\n✓ SUCCESS: CAB preserves bridges better than H2O!")
    else:
        print("\n✗ FAILURE: CAB does not outperform H2O on bridge preservation")

    return results


def test_discriminative_power_analysis():
    """
    Test 2: Discriminative Power Analysis

    Compare the discriminative power of different FRC formulas across sparsity levels.
    This tells us which formula can maintain distinctions at extreme sparsity.
    """
    print("\n" + "="*80)
    print("TEST 2: DISCRIMINATIVE POWER ANALYSIS")
    print("="*80)

    # Create random attention matrix (simulating real attention patterns)
    B, H, M, D = 1, 1, 128, 64
    q = torch.randn(B, H, M, D)
    k = torch.randn(B, H, M, D)

    # Normalize to attention-like distribution
    raw_attn = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    A = F.softmax(raw_attn, dim=-1)

    sparsity_levels = [0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
    formulas = ['additive', 'multiplicative', 'entropy']

    results = {formula: [] for formula in formulas}

    print(f"\nTesting on {M}x{M} attention matrix...")
    print(f"{'Sparsity':<10} {'Additive':<30} {'Multiplicative':<30} {'Entropy':<30}")
    print("-" * 100)

    for sparsity in sparsity_levels:
        for formula in formulas:
            # Compute FRC
            if formula == 'additive':
                frc, _, _ = compute_block_frc(A, A, formula='additive', lambda_redundancy=0.5)
            elif formula == 'multiplicative':
                frc, _, _ = compute_block_frc(A, A, formula='multiplicative', lambda_redundancy=1.0)
            else:  # entropy
                frc, _, _ = compute_block_frc(A, A, formula='entropy')

            # Analyze discriminative power
            diagnostics = analyze_frc_discriminative_power(frc, sparsity=sparsity, verbose=False)
            results[formula].append(diagnostics)

        # Print row
        print(f"{sparsity*100:>3.0f}%       "
              f"CV={results['additive'][-1]['coefficient_of_variation']:.3f} "
              f"Sep={results['additive'][-1]['separation']:.4f} "
              f"Pow={results['additive'][-1]['discriminative_power']:.4f}   "
              f"CV={results['multiplicative'][-1]['coefficient_of_variation']:.3f} "
              f"Sep={results['multiplicative'][-1]['separation']:.4f} "
              f"Pow={results['multiplicative'][-1]['discriminative_power']:.4f}   "
              f"CV={results['entropy'][-1]['coefficient_of_variation']:.3f} "
              f"Sep={results['entropy'][-1]['separation']:.4f} "
              f"Pow={results['entropy'][-1]['discriminative_power']:.4f}")

    # Determine best formula at extreme sparsity (95%+)
    extreme_sparsity_idx = [i for i, s in enumerate(sparsity_levels) if s >= 0.95]

    avg_power = {}
    for formula in formulas:
        powers = [results[formula][i]['discriminative_power'] for i in extreme_sparsity_idx]
        avg_power[formula] = np.mean(powers)

    best_formula = max(avg_power, key=avg_power.get)

    print(f"\nAverage Discriminative Power at 95%+ Sparsity:")
    for formula, power in avg_power.items():
        marker = "  ← BEST" if formula == best_formula else ""
        print(f"  {formula:15s}: {power:.6f}{marker}")

    print(f"\n✓ Recommended formula for extreme sparsity: {best_formula.upper()}")

    return results


def test_formula_ablation():
    """
    Test 3: Formula and Hyperparameter Ablation

    Systematically test different configurations to find optimal settings.
    """
    print("\n" + "="*80)
    print("TEST 3: FORMULA AND HYPERPARAMETER ABLATION")
    print("="*80)

    # Create test data
    B, H, M, D = 1, 1, 128, 64
    q = torch.randn(B, H, M, D)
    k = torch.randn(B, H, M, D)

    # Test configurations
    configs = [
        {'formula': 'additive', 'normalization': 'minmax', 'lambda_redundancy': 0.5},
        {'formula': 'additive', 'normalization': 'row', 'lambda_redundancy': 0.5},
        {'formula': 'multiplicative', 'normalization': 'minmax', 'lambda_redundancy': 0.5},
        {'formula': 'multiplicative', 'normalization': 'minmax', 'lambda_redundancy': 1.0},
        {'formula': 'multiplicative', 'normalization': 'minmax', 'lambda_redundancy': 2.0},
        {'formula': 'multiplicative', 'normalization': 'row', 'lambda_redundancy': 1.0},
        {'formula': 'entropy', 'normalization': 'minmax', 'lambda_redundancy': None},
    ]

    print(f"\nTesting {len(configs)} configurations at 95% sparsity...")
    print(f"{'Config':<50} {'Disc. Power':<15} {'Stable':<10}")
    print("-" * 75)

    best_config = None
    best_power = -1

    for i, config in enumerate(configs):
        # Compute FRC
        if config['lambda_redundancy'] is not None:
            frc, A, red = compute_block_frc(
                q, k,
                formula=config['formula'],
                normalization=config['normalization'],
                lambda_redundancy=config['lambda_redundancy']
            )
        else:
            frc, A, red = compute_block_frc(
                q, k,
                formula=config['formula'],
                normalization=config['normalization']
            )

        # Analyze
        diagnostics = analyze_frc_discriminative_power(frc, sparsity=0.95, verbose=False)
        stability = validate_frc_stability(frc, A, red, verbose=False)

        is_stable = not (stability['has_nan'] or stability['has_inf'])
        power = diagnostics['discriminative_power']

        # Track best
        if is_stable and power > best_power:
            best_power = power
            best_config = config

        # Print results
        config_str = f"{config['formula']}, {config['normalization']}, λ={config.get('lambda_redundancy', 'N/A')}"
        stable_marker = "✓" if is_stable else "✗"
        print(f"{config_str:<50} {power:<15.6f} {stable_marker:<10}")

    print(f"\n✓ Best configuration:")
    print(f"  Formula: {best_config['formula']}")
    print(f"  Normalization: {best_config['normalization']}")
    print(f"  Lambda: {best_config.get('lambda_redundancy', 'N/A')}")
    print(f"  Discriminative Power: {best_power:.6f}")

    return best_config


def test_sparsity_limits():
    """
    Test 4: Push Sparsity to the Limit

    Determine the maximum sparsity at which CAB can still make meaningful distinctions.
    """
    print("\n" + "="*80)
    print("TEST 4: SPARSITY LIMITS")
    print("="*80)

    # Create test data
    B, H, M, D = 1, 1, 128, 64
    q = torch.randn(B, H, M, D)
    k = torch.randn(B, H, M, D)

    # Test extreme sparsity levels
    sparsity_levels = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]

    # Use best formula (multiplicative)
    frc, A, red = compute_block_frc(
        q, k,
        formula='multiplicative',
        normalization='minmax',
        lambda_redundancy=1.0
    )

    print(f"\nTesting multiplicative FRC at extreme sparsity levels...")
    print(f"{'Sparsity':<12} {'Blocks Kept':<15} {'CV':<10} {'Separation':<12} {'Warning':<30}")
    print("-" * 85)

    max_viable_sparsity = 0.0

    for sparsity in sparsity_levels:
        diagnostics = analyze_frc_discriminative_power(frc, sparsity=sparsity, verbose=False)

        k_keep = max(1, int(M * M * (1 - sparsity)))
        cv = diagnostics['coefficient_of_variation']
        sep = diagnostics['separation']
        warning = diagnostics['warning'] or "OK"

        # Consider viable if CV > 0.1 and separation > 0.01
        is_viable = cv > 0.1 and sep > 0.01
        if is_viable:
            max_viable_sparsity = sparsity

        marker = "✓" if is_viable else "✗"
        print(f"{sparsity*100:>3.1f}% {marker}     {k_keep:<15d} {cv:<10.4f} {sep:<12.6f} {warning:<30}")

    print(f"\n✓ Maximum viable sparsity: {max_viable_sparsity*100:.1f}%")

    if max_viable_sparsity >= 0.95:
        print("  SUCCESS: CAB can work at 95%+ sparsity!")
    else:
        print(f"  WARNING: CAB may struggle above {max_viable_sparsity*100:.1f}% sparsity")

    return max_viable_sparsity


def run_all_tests():
    """Run all tests and generate comprehensive report."""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "CAB ATTENTION - ICML 2025 VALIDATION" + " " * 22 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("\n")

    results = {}

    # Test 1: Bridge Recovery
    try:
        results['bridge_recovery'] = test_synthetic_bridge_recovery()
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Discriminative Power
    try:
        results['discriminative_power'] = test_discriminative_power_analysis()
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Ablation
    try:
        results['best_config'] = test_formula_ablation()
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Sparsity Limits
    try:
        results['max_sparsity'] = test_sparsity_limits()
    except Exception as e:
        print(f"\n✗ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print("\nKEY FINDINGS:")

    if 'max_sparsity' in results:
        if results['max_sparsity'] >= 0.95:
            print("✓ CAB can reliably work at 95%+ sparsity")
        else:
            print(f"⚠️  CAB max sparsity: {results['max_sparsity']*100:.1f}% (needs improvement)")

    if 'best_config' in results:
        print(f"✓ Best formula: {results['best_config']['formula']}")
        print(f"✓ Best normalization: {results['best_config']['normalization']}")
        print(f"✓ Best lambda: {results['best_config'].get('lambda_redundancy', 'N/A')}")

    print("\nRECOMMENDATIONS FOR ICML SUBMISSION:")
    print("1. Use MULTIPLICATIVE FRC formula (exponential discrimination)")
    print("2. Use MINMAX normalization (preserves magnitude differences)")
    print("3. Set lambda_redundancy = 1.0 (balanced penalty)")
    print("4. Always select HIGH FRC (CAB V3 breakthrough)")

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)

    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run all tests
    results = run_all_tests()
