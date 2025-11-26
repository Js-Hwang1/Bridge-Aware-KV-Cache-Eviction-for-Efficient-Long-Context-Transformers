#!/usr/bin/env python3
"""
Quick kernel performance test for CAB-Attention.
Run: python benchmarks/quick_kernel_test.py
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def benchmark(name, fn, warmup=3, trials=10):
    """Simple benchmark."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(trials):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / trials * 1000
    print(f"  {name}: {elapsed:.3f} ms")
    return elapsed


def main():
    print("="*60)
    print("CAB-ATTENTION QUICK KERNEL TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test config: typical LLM inference
    B, H, N, D = 1, 32, 16384, 128
    block_size = 64
    M = N // block_size
    sparsity = 0.9
    
    print(f"\nConfig: B={B}, H={H}, N={N}, D={D}, block_size={block_size}")
    print(f"Coarsened: M={M} blocks")
    
    # Create test tensors
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    
    print("\n--- Coarsening ---")
    from cab_attention.kernels.coarsening import coarsen_qk_max_l2, coarsen_qk_max_l2_pytorch
    
    triton_time = benchmark("Triton", lambda: coarsen_qk_max_l2(q, k, block_size))
    pytorch_time = benchmark("PyTorch", lambda: coarsen_qk_max_l2_pytorch(q, k, block_size))
    print(f"  Speedup: {pytorch_time/triton_time:.2f}x")
    
    print("\n--- FRC Computation ---")
    from cab_attention.kernels.frc_kernel import compute_block_frc, generate_block_mask
    
    q_coarse, k_coarse = coarsen_qk_max_l2(q, k, block_size)
    q_c_f = q_coarse.float()
    k_c_f = k_coarse.float()
    
    benchmark("FRC (full)", lambda: compute_block_frc(q_c_f, k_c_f))
    
    print("\n--- Mask Generation ---")
    frc_scores, _, _ = compute_block_frc(q_c_f, k_c_f)
    mag_scores = torch.randn_like(frc_scores)
    
    benchmark("Pure FRC (CAB V3)", lambda: generate_block_mask(frc_scores, sparsity=sparsity))
    benchmark("Hybrid 50/50 (CAB V4)", lambda: generate_block_mask(
        frc_scores, sparsity=sparsity, magnitude_scores=mag_scores, magnitude_ratio=0.5))
    
    print("\n--- End-to-End Pipeline ---")
    def full_pipeline():
        qc, kc = coarsen_qk_max_l2(q, k, block_size)
        frc, _, _ = compute_block_frc(qc.float(), kc.float())
        mask = generate_block_mask(frc, sparsity=sparsity)
        return mask
    
    total_time = benchmark("Full CAB Pipeline", full_pipeline)
    
    print("\n--- Summary ---")
    print(f"Total overhead for N={N}: {total_time:.2f} ms")
    print(f"Amortized per-head: {total_time/H:.3f} ms")
    
    # Reference: FlashAttention takes ~2-3ms for similar config
    print(f"\nFor reference: FlashAttention-2 for this config: ~2-3 ms")
    print(f"CAB overhead ratio: {total_time/2.5:.1f}x attention time")
    
    if total_time < 10:
        print("\n✅ Overhead acceptable for deployment!")
    elif total_time < 50:
        print("\n⚠️  Moderate overhead - OK for research, optimize for production")
    else:
        print("\n❌ High overhead - needs optimization!")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

