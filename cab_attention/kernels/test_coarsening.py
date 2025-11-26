"""
Test and benchmark suite for coarsening kernels.

Tests:
1. Correctness: Triton kernel matches PyTorch reference
2. Performance: Benchmark Triton vs PyTorch across various shapes
3. Memory: Profile memory usage
"""

import torch
import time
from coarsening import coarsen_qk_max_l2, coarsen_qk_max_l2_pytorch


def test_correctness(B=2, H=8, N=1024, D=128, block_size=64):
    """
    Test that Triton kernel produces identical results to PyTorch reference.
    """
    print(f"\n{'='*60}")
    print(f"CORRECTNESS TEST: B={B}, H={H}, N={N}, D={D}, block_size={block_size}")
    print(f"{'='*60}")

    # Create random inputs
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    # Run PyTorch reference
    q_pytorch, k_pytorch = coarsen_qk_max_l2_pytorch(q.clone(), k.clone(), block_size)

    # Run Triton kernel
    q_triton, k_triton = coarsen_qk_max_l2(q.clone(), k.clone(), block_size)

    # Compare results
    q_match = torch.allclose(q_triton, q_pytorch, rtol=1e-5, atol=1e-5)
    k_match = torch.allclose(k_triton, k_pytorch, rtol=1e-5, atol=1e-5)

    if q_match and k_match:
        print("✅ PASS: Triton output matches PyTorch reference")

        # Compute max absolute difference
        q_max_diff = (q_triton - q_pytorch).abs().max().item()
        k_max_diff = (k_triton - k_pytorch).abs().max().item()
        print(f"   Max absolute difference (Q): {q_max_diff:.2e}")
        print(f"   Max absolute difference (K): {k_max_diff:.2e}")
        return True
    else:
        print("❌ FAIL: Output mismatch!")
        if not q_match:
            q_diff = (q_triton - q_pytorch).abs()
            print(f"   Q max diff: {q_diff.max().item():.2e}")
            print(f"   Q mean diff: {q_diff.mean().item():.2e}")
        if not k_match:
            k_diff = (k_triton - k_pytorch).abs()
            print(f"   K max diff: {k_diff.max().item():.2e}")
            print(f"   K mean diff: {k_diff.mean().item():.2e}")
        return False


def benchmark(B=2, H=8, N=8192, D=128, block_size=64, n_warmup=10, n_iter=100):
    """
    Benchmark Triton kernel vs PyTorch reference.
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: B={B}, H={H}, N={N}, D={D}, block_size={block_size}")
    print(f"{'='*60}")

    # Create random inputs
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(n_warmup):
        _ = coarsen_qk_max_l2_pytorch(q, k, block_size)
        _ = coarsen_qk_max_l2(q, k, block_size)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = coarsen_qk_max_l2_pytorch(q, k, block_size)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iter

    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = coarsen_qk_max_l2(q, k, block_size)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iter

    # Calculate speedup
    speedup = pytorch_time / triton_time

    # Calculate effective bandwidth
    # Input: 2 tensors (Q, K) of size B*H*N*D * 4 bytes
    # Output: 2 tensors of size B*H*M*D * 4 bytes where M = N/block_size
    M = (N + block_size - 1) // block_size
    input_bytes = 2 * B * H * N * D * 4
    output_bytes = 2 * B * H * M * D * 4
    total_bytes = input_bytes + output_bytes

    pytorch_bandwidth = total_bytes / pytorch_time / 1e9  # GB/s
    triton_bandwidth = total_bytes / triton_time / 1e9    # GB/s

    print(f"PyTorch:  {pytorch_time*1000:.3f} ms  ({pytorch_bandwidth:.1f} GB/s)")
    print(f"Triton:   {triton_time*1000:.3f} ms  ({triton_bandwidth:.1f} GB/s)")
    print(f"Speedup:  {speedup:.2f}x")

    if speedup > 1.0:
        print("✅ Triton is faster!")
    else:
        print("⚠️  PyTorch is faster - kernel needs optimization")

    return {
        'pytorch_time': pytorch_time,
        'triton_time': triton_time,
        'speedup': speedup,
        'pytorch_bandwidth': pytorch_bandwidth,
        'triton_bandwidth': triton_bandwidth
    }


def benchmark_sweep():
    """
    Comprehensive benchmark across multiple configurations.
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE BENCHMARK SWEEP")
    print("="*60)

    configs = [
        # (B, H, N, D, block_size)
        # Small sequences
        (1, 8, 1024, 128, 64),
        (2, 8, 2048, 128, 64),

        # Medium sequences (typical LLM inference)
        (1, 32, 8192, 128, 64),
        (2, 32, 16384, 128, 64),

        # Long sequences (target for CAB)
        (1, 32, 32768, 128, 64),
        (1, 32, 65536, 128, 64),

        # Different block sizes
        (1, 32, 16384, 128, 32),
        (1, 32, 16384, 128, 128),

        # Different head dimensions
        (1, 32, 16384, 64, 64),
        (1, 32, 16384, 256, 64),
    ]

    results = []
    for B, H, N, D, block_size in configs:
        try:
            result = benchmark(B, H, N, D, block_size, n_warmup=5, n_iter=50)
            result['config'] = (B, H, N, D, block_size)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Config (B,H,N,D,BS)':<30} {'Speedup':>10} {'Triton GB/s':>12}")
    print("-"*60)
    for r in results:
        B, H, N, D, bs = r['config']
        config_str = f"({B},{H},{N},{D},{bs})"
        print(f"{config_str:<30} {r['speedup']:>9.2f}x {r['triton_bandwidth']:>11.1f}")

    # Average speedup
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print("-"*60)
    print(f"{'Average':<30} {avg_speedup:>9.2f}x")


def test_edge_cases():
    """
    Test edge cases to ensure robustness.
    """
    print("\n" + "="*60)
    print("EDGE CASE TESTS")
    print("="*60)

    edge_cases = [
        # (B, H, N, D, block_size, description)
        (1, 1, 64, 64, 32, "Minimal batch/heads"),
        (1, 8, 63, 128, 64, "N not divisible by block_size"),
        (1, 8, 100, 127, 64, "D not power of 2"),
        (4, 32, 4096, 128, 32, "Large batch/heads"),
    ]

    all_passed = True
    for B, H, N, D, block_size, desc in edge_cases:
        print(f"\nTest: {desc}")
        try:
            passed = test_correctness(B, H, N, D, block_size)
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ FAILED with error: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ All edge cases passed!")
    else:
        print("\n❌ Some edge cases failed")


if __name__ == "__main__":
    print("CAB-Attention Coarsening Kernel Test Suite")
    print("="*60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Run tests
    print("\n" + "="*60)
    print("PHASE 1: CORRECTNESS TESTS")
    print("="*60)
    test_correctness(B=2, H=8, N=1024, D=128, block_size=64)

    print("\n" + "="*60)
    print("PHASE 2: EDGE CASES")
    print("="*60)
    test_edge_cases()

    print("\n" + "="*60)
    print("PHASE 3: PERFORMANCE BENCHMARKS")
    print("="*60)

    # Quick benchmark
    benchmark(B=1, H=32, N=16384, D=128, block_size=64)

    # Full sweep (comment out for quick testing)
    response = input("\nRun comprehensive benchmark sweep? (y/n): ")
    if response.lower() == 'y':
        benchmark_sweep()

    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)
