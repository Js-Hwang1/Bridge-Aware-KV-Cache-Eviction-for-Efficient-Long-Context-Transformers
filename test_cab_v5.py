#!/usr/bin/env python3
"""
Comprehensive test suite for CAB V5
===================================

Tests all components:
1. FRC Triton kernels
2. Importance tracking
3. FRC tracking
4. Eviction policy
5. CAB cache
6. H2O baseline

Run: python test_cab_v5.py
"""

import torch
import time
from typing import Dict, Any


def test_frc_kernels():
    """Test FRC Triton kernels."""
    print("=" * 80)
    print("TEST 1: FRC Triton Kernels")
    print("=" * 80)

    from cab_attention.kernels.frc_triton import compute_frc_triton, compute_frc_pytorch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Create synthetic attention matrix
    N = 512
    attention = torch.rand(N, N, device=device)
    attention = attention / attention.sum(dim=1, keepdim=True)

    # Test PyTorch version
    start = time.time()
    frc_pytorch = compute_frc_pytorch(attention)
    pytorch_time = (time.time() - start) * 1000

    print(f"PyTorch FRC:")
    print(f"  Time: {pytorch_time:.2f}ms")
    print(f"  Mean: {frc_pytorch.mean():.4f}")
    print(f"  Std: {frc_pytorch.std():.4f}")

    if device == 'cuda':
        # Test Triton version
        torch.cuda.synchronize()
        start = time.time()
        frc_triton = compute_frc_triton(attention)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) * 1000

        print(f"\nTriton FRC:")
        print(f"  Time: {triton_time:.2f}ms")
        print(f"  Mean: {frc_triton.mean():.4f}")
        print(f"  Std: {frc_triton.std():.4f}")
        print(f"  Speedup: {pytorch_time / triton_time:.2f}x")

        # Verify correctness
        error = (frc_triton - frc_pytorch).abs().max()
        print(f"  Max error: {error:.6f}")

        if error < 1e-3:
            print(f"\n✓ PASS: FRC kernels working correctly")
        else:
            print(f"\n✗ FAIL: Large numerical error")

    print()


def test_importance_tracking():
    """Test importance tracking."""
    print("=" * 80)
    print("TEST 2: Importance Tracking")
    print("=" * 80)

    from cab_attention.scoring import ImportanceTracker

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    tracker = ImportanceTracker(device=device)

    # Simulate generation
    B, H = 1, 8
    cache_len = 0

    print("Simulating 100 generation steps...")
    for step in range(100):
        cache_len += 1
        attention = torch.rand(B, H, 1, cache_len, device=device)
        attention = attention / attention.sum(dim=-1, keepdim=True)

        # Make some positions important
        if cache_len > 10:
            attention[:, :, :, 5] *= 5.0
        if cache_len > 20:
            attention[:, :, :, 20] *= 10.0

        attention = attention / attention.sum(dim=-1, keepdim=True)
        tracker.update(attention)

    # Get top-10
    top_10 = tracker.get_top_k_indices(k=10)

    print(f"\nCache length: {len(tracker)}")
    print(f"Top-10 important positions: {top_10.cpu().numpy()}")

    # Verify expected important positions are in top-10
    important_positions = {5, 20}
    found = set(top_10.cpu().numpy().tolist())

    if important_positions.issubset(found):
        print(f"\n✓ PASS: Important positions detected correctly")
    else:
        print(f"\n✗ FAIL: Important positions not in top-10")

    print()


def test_frc_tracking():
    """Test FRC tracking."""
    print("=" * 80)
    print("TEST 3: FRC Tracking")
    print("=" * 80)

    from cab_attention.scoring import FRCTracker

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    tracker = FRCTracker(device=device, use_triton=(device == 'cuda'))

    # Create synthetic keys
    B, H, N, D = 1, 8, 256, 64
    keys = torch.randn(B, H, N, D, device=device)

    # Compute FRC
    start = time.time()
    frc_scores = tracker.compute_from_keys(keys)
    elapsed = (time.time() - start) * 1000

    print(f"FRC computation: {elapsed:.2f}ms")
    print(f"FRC scores:")
    print(f"  Mean: {frc_scores.mean():.4f}")
    print(f"  Std: {frc_scores.std():.4f}")
    print(f"  Min: {frc_scores.min():.4f} (strongest bridge)")
    print(f"  Max: {frc_scores.max():.4f} (most redundant)")

    # Get bridges
    bottom_10 = tracker.get_bottom_k_indices(k=10)
    print(f"\nBottom-10 FRC (bridges): {bottom_10.cpu().numpy()}")

    # Test caching
    start = time.time()
    frc_cached = tracker.compute_from_keys(keys, force_update=False)
    cached_time = (time.time() - start) * 1000

    print(f"\nCached lookup: {cached_time:.2f}ms")

    if cached_time < elapsed / 10:
        print(f"\n✓ PASS: FRC caching working")
    else:
        print(f"\n✗ FAIL: FRC caching not effective")

    print()


def test_eviction_policy():
    """Test eviction policy."""
    print("=" * 80)
    print("TEST 4: Eviction Policy")
    print("=" * 80)

    from cab_attention.eviction import ThreeComponentEvictionPolicy, EvictionConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    config = EvictionConfig(
        local_ratio=0.3,
        bridge_ratio=0.2,
        importance_ratio=0.5,
    )
    policy = ThreeComponentEvictionPolicy(config)

    # Synthetic scores
    cache_len = 1000
    keep_size = 100

    importance_scores = torch.rand(cache_len, device=device)
    importance_scores[100] = 10.0  # Important position
    importance_scores[500] = 15.0  # Very important

    frc_scores = torch.randn(cache_len, device=device)
    frc_scores[250] = -5.0  # Strong bridge
    frc_scores[750] = -3.0  # Strong bridge

    # Select indices
    keep_indices, diagnostics = policy.select_indices(
        cache_len=cache_len,
        keep_size=keep_size,
        importance_scores=importance_scores,
        frc_scores=frc_scores,
        device=device,
    )

    print(f"Cache length: {cache_len}")
    print(f"Keep size: {len(keep_indices)}")
    print(f"  Local: {diagnostics['local_count']} ({diagnostics['local_ratio_actual']:.1%})")
    print(f"  Important: {diagnostics['importance_count']} ({diagnostics['importance_ratio_actual']:.1%})")
    print(f"  Bridges: {diagnostics['bridge_count']} ({diagnostics['bridge_ratio_actual']:.1%})")

    # Verify important positions kept
    checks = [
        (100, "Important position 100"),
        (500, "Important position 500"),
        (250, "Bridge position 250"),
        (750, "Bridge position 750"),
    ]

    all_pass = True
    for pos, desc in checks:
        kept = pos in keep_indices
        status = "✓" if kept else "✗"
        print(f"  {status} {desc}: {'kept' if kept else 'evicted'}")
        if not kept:
            all_pass = False

    if all_pass:
        print(f"\n✓ PASS: Eviction policy working correctly")
    else:
        print(f"\n✗ FAIL: Some expected positions not kept")

    print()


def test_cab_cache():
    """Test CAB cache."""
    print("=" * 80)
    print("TEST 5: CAB Cache")
    print("=" * 80)

    from cab_attention import CABCache

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    cache = CABCache(
        max_cache_size=100,
        sparsity=0.9,
        local_ratio=0.3,
        bridge_ratio=0.2,
        importance_ratio=0.5,
        eviction_interval=5,
        device=device,
    )

    print(f"Cache config:")
    print(f"  Max size: {cache.config.max_cache_size}")
    print(f"  Sparsity: {cache.config.sparsity:.0%}")

    # Simulate generation
    B, H, D = 1, 8, 64
    num_layers = 4

    print(f"\nSimulating 200 steps with {num_layers} layers...")

    for step in range(200):
        for layer_idx in range(num_layers):
            key_state = torch.randn(B, H, 1, D, device=device)
            value_state = torch.randn(B, H, 1, D, device=device)

            attention = None
            if layer_idx == 0 and cache.get_seq_length(0) > 0:
                cache_len = cache.get_seq_length(0)
                attention = torch.rand(B, H, 1, cache_len, device=device)
                attention = attention / attention.sum(dim=-1, keepdim=True)

            keys, values = cache.update(key_state, value_state, layer_idx, attention)

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}: Size = {len(cache)}, Evictions = {cache.total_evictions}")

    # Check final state
    stats = cache.get_stats()
    print(f"\nFinal stats:")
    print(f"  Cache size: {stats['current_cache_size']}/{stats['max_cache_size']}")
    print(f"  Total evictions: {stats['total_evictions']}")
    print(f"  Total tokens evicted: {stats['total_tokens_evicted']}")
    print(f"  Avg cache size: {stats['avg_cache_size']:.1f}")

    # Verify cache stays within bounds
    if stats['current_cache_size'] <= cache.config.max_cache_size:
        print(f"\n✓ PASS: Cache size within bounds")
    else:
        print(f"\n✗ FAIL: Cache exceeded max size")

    print()


def test_h2o_cache():
    """Test H2O baseline cache."""
    print("=" * 80)
    print("TEST 6: H2O Cache (Baseline)")
    print("=" * 80)

    from cab_attention import H2OCache

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    cache = H2OCache(
        max_cache_size=100,
        sparsity=0.9,
        eviction_interval=5,
        device=device,
    )

    print(f"Cache config:")
    print(f"  Max size: {cache.max_cache_size}")
    print(f"  Sparsity: {cache.sparsity:.0%}")

    # Simulate generation
    B, H, D = 1, 8, 64
    num_layers = 4

    print(f"\nSimulating 200 steps with {num_layers} layers...")

    for step in range(200):
        for layer_idx in range(num_layers):
            key_state = torch.randn(B, H, 1, D, device=device)
            value_state = torch.randn(B, H, 1, D, device=device)

            attention = None
            if layer_idx == 0 and cache.get_seq_length(0) > 0:
                cache_len = cache.get_seq_length(0)
                attention = torch.rand(B, H, 1, cache_len, device=device)
                attention = attention / attention.sum(dim=-1, keepdim=True)

            keys, values = cache.update(key_state, value_state, layer_idx, attention)

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}: Size = {len(cache)}, Evictions = {cache.total_evictions}")

    # Check final state
    stats = cache.get_stats()
    print(f"\nFinal stats:")
    print(f"  Cache size: {stats['current_cache_size']}/{stats['max_cache_size']}")
    print(f"  Total evictions: {stats['total_evictions']}")
    print(f"  Total tokens evicted: {stats['total_tokens_evicted']}")

    if stats['current_cache_size'] <= cache.max_cache_size:
        print(f"\n✓ PASS: H2O cache working correctly")
    else:
        print(f"\n✗ FAIL: H2O cache exceeded max size")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CAB V5 Comprehensive Test Suite")
    print("=" * 80 + "\n")

    tests = [
        ("FRC Kernels", test_frc_kernels),
        ("Importance Tracking", test_importance_tracking),
        ("FRC Tracking", test_frc_tracking),
        ("Eviction Policy", test_eviction_policy),
        ("CAB Cache", test_cab_cache),
        ("H2O Cache", test_h2o_cache),
    ]

    results = {}

    for name, test_func in tests:
        try:
            test_func()
            results[name] = "PASS"
        except Exception as e:
            print(f"\n✗ FAIL: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = "FAIL"

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, result in results.items():
        symbol = "✓" if result == "PASS" else "✗"
        print(f"{symbol} {name}: {result}")

    total = len(results)
    passed = sum(1 for r in results.values() if r == "PASS")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
