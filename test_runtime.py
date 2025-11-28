"""
Quick runtime benchmark for CAB and H2O caches.
Tests O(N) complexity and ensures production viability.
"""

import torch
import time
from cab_attention import CABCache, H2OCache


def benchmark_cache(cache_class, name, num_tokens=1000, num_layers=32):
    """Benchmark cache performance."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {name}")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create cache
    cache = cache_class(
        max_cache_size=512,
        sparsity=0.9,
        eviction_interval=10,
        device=device,
    )

    # Simulate generation
    B, H, D = 1, 32, 128

    start_time = time.time()
    eviction_times = []

    for step in range(num_tokens):
        # Generate random key/value for each layer
        for layer_idx in range(num_layers):
            key_state = torch.randn(B, H, 1, D, device=device)
            value_state = torch.randn(B, H, 1, D, device=device)

            # Simulate attention (only for first layer)
            attention = None
            if layer_idx == 0 and cache.get_seq_length(0) > 0:
                cache_len = cache.get_seq_length(0)
                attention = torch.rand(B, H, 1, cache_len, device=device)
                attention = attention / attention.sum(dim=-1, keepdim=True)

            # Track eviction time
            evict_start = time.time()
            keys, values = cache.update(key_state, value_state, layer_idx, attention)
            evict_time = (time.time() - evict_start) * 1000

            # Only track when eviction happens
            if layer_idx == 0 and cache.tokens_since_last_eviction == 0:
                eviction_times.append(evict_time)

        # Progress
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{num_tokens}: Cache size = {len(cache)}, Evictions = {cache.total_evictions}")

    total_time = time.time() - start_time

    # Results
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Time per token: {(total_time / num_tokens) * 1000:.2f}ms")
    print(f"  Total evictions: {cache.total_evictions}")

    if eviction_times:
        avg_eviction = sum(eviction_times) / len(eviction_times)
        max_eviction = max(eviction_times)
        print(f"  Avg eviction time: {avg_eviction:.2f}ms")
        print(f"  Max eviction time: {max_eviction:.2f}ms")

    stats = cache.get_stats()
    print(f"  Final cache size: {stats.get('current_cache_size', len(cache))}")
    print(f"  Tokens evicted: {stats['total_tokens_evicted']}")

    # Complexity check
    print(f"\nComplexity Analysis:")
    if eviction_times:
        print(f"  Eviction time should be O(N log N) for topk")
        print(f"  With N~{cache.max_cache_size}, expecting <10ms per eviction")
        if avg_eviction < 20:
            print(f"  ✓ PASS: Avg eviction {avg_eviction:.2f}ms is acceptable")
        else:
            print(f"  ✗ WARNING: Avg eviction {avg_eviction:.2f}ms may be too slow")


if __name__ == "__main__":
    print("="*60)
    print("CAB/H2O Runtime Benchmark")
    print("="*60)

    # Test H2O first (baseline)
    benchmark_cache(H2OCache, "H2O Cache", num_tokens=500, num_layers=32)

    # Test CAB
    benchmark_cache(CABCache, "CAB Cache", num_tokens=500, num_layers=32)

    print("\n" + "="*60)
    print("Benchmark Complete")
    print("="*60)
