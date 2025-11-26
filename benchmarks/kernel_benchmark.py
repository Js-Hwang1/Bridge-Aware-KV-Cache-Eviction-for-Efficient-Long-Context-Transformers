#!/usr/bin/env python3
"""
Comprehensive Kernel Benchmark for CAB-Attention (ICML 2025)

Benchmarks:
1. Coarsening kernel (Triton vs PyTorch)
2. FRC computation
3. Mask generation  
4. End-to-end CAB pipeline

Run on A100:
    python benchmarks/kernel_benchmark.py
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cab_attention.kernels.coarsening import coarsen_qk_max_l2, coarsen_qk_max_l2_pytorch
from cab_attention.kernels.frc_kernel import compute_block_frc, generate_block_mask


def benchmark_fn(fn, *args, warmup=5, trials=20, **kwargs):
    """Benchmark a function with warmup and multiple trials."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def benchmark_coarsening():
    """Benchmark coarsening kernel."""
    print("\n" + "="*70)
    print("COARSENING KERNEL BENCHMARK")
    print("="*70)
    
    configs = [
        # (B, H, N, D, block_size)
        (1, 8, 1024, 128, 64),      # Small
        (1, 8, 4096, 128, 64),      # Medium
        (1, 8, 16384, 128, 64),     # Large
        (1, 32, 16384, 128, 64),    # Many heads
        (1, 8, 32768, 128, 64),     # Very large
        (1, 8, 65536, 128, 64),     # 64K context
        (1, 8, 131072, 128, 64),    # 128K context
        (2, 8, 16384, 128, 64),     # Batch=2
        (1, 8, 16384, 256, 64),     # Large D
    ]
    
    print(f"\n{'Config':<35} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 75)
    
    results = []
    for B, H, N, D, block_size in configs:
        try:
            q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
            k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
            
            # Triton
            triton_stats = benchmark_fn(coarsen_qk_max_l2, q, k, block_size)
            
            # PyTorch
            pytorch_stats = benchmark_fn(coarsen_qk_max_l2_pytorch, q, k, block_size)
            
            speedup = pytorch_stats['mean_ms'] / triton_stats['mean_ms']
            
            config_str = f"B={B}, H={H}, N={N}, D={D}"
            print(f"{config_str:<35} {triton_stats['mean_ms']:<15.3f} {pytorch_stats['mean_ms']:<15.3f} {speedup:<10.2f}x")
            
            results.append({
                'config': (B, H, N, D, block_size),
                'triton_ms': triton_stats['mean_ms'],
                'pytorch_ms': pytorch_stats['mean_ms'],
                'speedup': speedup,
            })
            
            del q, k
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{'B='+str(B)+', H='+str(H)+', N='+str(N)+', D='+str(D):<35} ERROR: {e}")
    
    return results


def benchmark_frc():
    """Benchmark FRC computation."""
    print("\n" + "="*70)
    print("FRC COMPUTATION BENCHMARK")
    print("="*70)
    
    configs = [
        # (B, H, M, D) - M is number of blocks after coarsening
        (1, 8, 64, 128),      # N=4K, block_size=64
        (1, 8, 256, 128),     # N=16K
        (1, 8, 512, 128),     # N=32K  
        (1, 8, 1024, 128),    # N=64K
        (1, 8, 2048, 128),    # N=128K
        (1, 32, 512, 128),    # Many heads
        (2, 8, 512, 128),     # Batch=2
    ]
    
    print(f"\n{'Config':<35} {'Affinity (ms)':<15} {'Redundancy (ms)':<15} {'FRC Total (ms)':<15}")
    print("-" * 80)
    
    results = []
    for B, H, M, D in configs:
        try:
            q_coarse = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16)
            k_coarse = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16)
            
            # Benchmark affinity computation
            def compute_affinity():
                scale = 1.0 / (D ** 0.5)
                return torch.matmul(q_coarse, k_coarse.transpose(-2, -1)) * scale
            
            affinity_stats = benchmark_fn(compute_affinity)
            
            # Benchmark redundancy computation (A @ A)
            A = compute_affinity()
            def compute_redundancy():
                return torch.matmul(A, A)
            
            redundancy_stats = benchmark_fn(compute_redundancy)
            
            # Full FRC
            frc_stats = benchmark_fn(compute_block_frc, q_coarse, k_coarse)
            
            config_str = f"B={B}, H={H}, M={M}, D={D}"
            print(f"{config_str:<35} {affinity_stats['mean_ms']:<15.3f} {redundancy_stats['mean_ms']:<15.3f} {frc_stats['mean_ms']:<15.3f}")
            
            results.append({
                'config': (B, H, M, D),
                'affinity_ms': affinity_stats['mean_ms'],
                'redundancy_ms': redundancy_stats['mean_ms'],
                'frc_total_ms': frc_stats['mean_ms'],
            })
            
            del q_coarse, k_coarse, A
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{'B='+str(B)+', H='+str(H)+', M='+str(M)+', D='+str(D):<35} ERROR: {e}")
    
    return results


def benchmark_mask_generation():
    """Benchmark mask generation."""
    print("\n" + "="*70)
    print("MASK GENERATION BENCHMARK")
    print("="*70)
    
    configs = [
        # (B, H, M, sparsity)
        (1, 8, 256, 0.9),
        (1, 8, 512, 0.9),
        (1, 8, 1024, 0.9),
        (1, 8, 2048, 0.9),
        (1, 32, 512, 0.9),
        (1, 8, 512, 0.95),
        (1, 8, 512, 0.99),
    ]
    
    print(f"\n{'Config':<40} {'Pure FRC (ms)':<15} {'Hybrid 50/50 (ms)':<18}")
    print("-" * 75)
    
    results = []
    for B, H, M, sparsity in configs:
        try:
            frc_scores = torch.randn(B, H, M, M, device='cuda', dtype=torch.float32)
            magnitude_scores = torch.randn(B, H, M, M, device='cuda', dtype=torch.float32)
            
            # Pure FRC (CAB V3)
            pure_frc_stats = benchmark_fn(
                generate_block_mask, frc_scores, sparsity=sparsity
            )
            
            # Hybrid (CAB V4) - Note: this has Python loops!
            hybrid_stats = benchmark_fn(
                generate_block_mask, frc_scores, sparsity=sparsity,
                magnitude_scores=magnitude_scores, magnitude_ratio=0.5
            )
            
            config_str = f"B={B}, H={H}, M={M}, sparsity={sparsity}"
            print(f"{config_str:<40} {pure_frc_stats['mean_ms']:<15.3f} {hybrid_stats['mean_ms']:<18.3f}")
            
            results.append({
                'config': (B, H, M, sparsity),
                'pure_frc_ms': pure_frc_stats['mean_ms'],
                'hybrid_ms': hybrid_stats['mean_ms'],
            })
            
            del frc_scores, magnitude_scores
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Config {(B, H, M, sparsity)} ERROR: {e}")
    
    return results


def benchmark_end_to_end():
    """Benchmark full CAB pipeline."""
    print("\n" + "="*70)
    print("END-TO-END CAB PIPELINE BENCHMARK")
    print("="*70)
    
    configs = [
        # (B, H, N, D, block_size, sparsity)
        (1, 8, 4096, 128, 64, 0.9),
        (1, 8, 16384, 128, 64, 0.9),
        (1, 8, 32768, 128, 64, 0.9),
        (1, 8, 65536, 128, 64, 0.9),
        (1, 8, 16384, 128, 64, 0.95),
        (1, 8, 16384, 128, 64, 0.99),
    ]
    
    print(f"\n{'Config':<45} {'Coarsen':<12} {'FRC':<12} {'Mask':<12} {'Total (ms)':<12}")
    print("-" * 95)
    
    results = []
    for B, H, N, D, block_size, sparsity in configs:
        try:
            q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
            k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
            
            # Full pipeline
            def full_pipeline():
                q_coarse, k_coarse = coarsen_qk_max_l2(q, k, block_size)
                frc_scores, _, _ = compute_block_frc(q_coarse.float(), k_coarse.float())
                mask = generate_block_mask(frc_scores, sparsity=sparsity)
                return mask
            
            # Individual components
            coarsen_stats = benchmark_fn(coarsen_qk_max_l2, q, k, block_size)
            
            q_coarse, k_coarse = coarsen_qk_max_l2(q, k, block_size)
            frc_stats = benchmark_fn(compute_block_frc, q_coarse.float(), k_coarse.float())
            
            frc_scores, _, _ = compute_block_frc(q_coarse.float(), k_coarse.float())
            mask_stats = benchmark_fn(generate_block_mask, frc_scores, sparsity=sparsity)
            
            total_stats = benchmark_fn(full_pipeline)
            
            config_str = f"N={N}, sp={sparsity}"
            print(f"B={B}, H={H}, {config_str:<30} {coarsen_stats['mean_ms']:<12.3f} {frc_stats['mean_ms']:<12.3f} {mask_stats['mean_ms']:<12.3f} {total_stats['mean_ms']:<12.3f}")
            
            results.append({
                'config': (B, H, N, D, block_size, sparsity),
                'coarsen_ms': coarsen_stats['mean_ms'],
                'frc_ms': frc_stats['mean_ms'],
                'mask_ms': mask_stats['mean_ms'],
                'total_ms': total_stats['mean_ms'],
            })
            
            del q, k, q_coarse, k_coarse, frc_scores
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Config {(B, H, N, D, block_size, sparsity)} ERROR: {e}")
    
    return results


def analyze_bottlenecks(coarsen_results, frc_results, mask_results, e2e_results):
    """Analyze performance bottlenecks."""
    print("\n" + "="*70)
    print("BOTTLENECK ANALYSIS")
    print("="*70)
    
    if e2e_results:
        # Find where time is spent
        example = e2e_results[-1]  # Take largest config
        total = example['total_ms']
        coarsen_pct = example['coarsen_ms'] / total * 100
        frc_pct = example['frc_ms'] / total * 100
        mask_pct = example['mask_ms'] / total * 100
        
        print(f"\nTime Distribution (N={example['config'][2]}):")
        print(f"  Coarsening: {example['coarsen_ms']:.2f}ms ({coarsen_pct:.1f}%)")
        print(f"  FRC:        {example['frc_ms']:.2f}ms ({frc_pct:.1f}%)")
        print(f"  Masking:    {example['mask_ms']:.2f}ms ({mask_pct:.1f}%)")
        print(f"  Total:      {total:.2f}ms")
    
    # Check Triton vs PyTorch speedup
    if coarsen_results:
        avg_speedup = sum(r['speedup'] for r in coarsen_results) / len(coarsen_results)
        print(f"\nCoarsening Triton Speedup: {avg_speedup:.2f}x average")
        
        if avg_speedup < 2.0:
            print("  ⚠️  WARNING: Triton speedup is low! Check kernel optimization.")
        elif avg_speedup < 5.0:
            print("  ⚡ Moderate speedup. Room for improvement.")
        else:
            print("  ✅ Good speedup!")
    
    # Check if mask generation is slow
    if mask_results:
        for r in mask_results:
            if r['hybrid_ms'] > 10:
                print(f"\n  ⚠️  WARNING: Hybrid mask generation is slow ({r['hybrid_ms']:.1f}ms)")
                print("     Python loops in generate_block_mask() are the bottleneck!")
                break


def main():
    print("="*70)
    print("CAB-ATTENTION KERNEL BENCHMARK")
    print("="*70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    try:
        import triton
        print(f"Triton Version: {triton.__version__}")
    except ImportError:
        print("WARNING: Triton not available")
    
    # Run benchmarks
    coarsen_results = benchmark_coarsening()
    frc_results = benchmark_frc()
    mask_results = benchmark_mask_generation()
    e2e_results = benchmark_end_to_end()
    
    # Analyze
    analyze_bottlenecks(coarsen_results, frc_results, mask_results, e2e_results)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

