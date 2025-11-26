"""
Benchmark Script: Coarse FRC Predictor Performance

Tests the predictor overhead for various sequence lengths up to 128k tokens.
Goal: Predictor latency must be < 5ms for N=128k to be viable.
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('..')

from cab_attention.coarse_predictor import CoarseCurvaturePredictor


def benchmark_predictor(
    sequence_lengths: list[int],
    block_size: int = 64,
    sparsity: float = 0.95,
    num_heads: int = 8,
    head_dim: int = 128,
    num_trials: int = 10,
    warmup: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Benchmarks the coarse FRC predictor across different sequence lengths.

    Args:
        sequence_lengths: List of N values to test
        block_size: Block size for coarsening
        sparsity: Target sparsity ratio
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_trials: Number of timing runs
        warmup: Number of warmup runs
        device: Device to run on

    Returns:
        DataFrame with benchmark results
    """
    print("=" * 80)
    print("CAB-ATTENTION PREDICTOR BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Block Size: {block_size}")
    print(f"Sparsity: {sparsity:.1%}")
    print(f"Num Heads: {num_heads}")
    print(f"Head Dim: {head_dim}")
    print(f"Trials: {num_trials} (warmup: {warmup})")
    print("=" * 80)

    # Initialize predictor
    predictor = CoarseCurvaturePredictor(
        block_size=block_size,
        sparsity=sparsity,
        use_triton=False,  # Use PyTorch for now
    ).to(device)

    results = []

    for N in tqdm(sequence_lengths, desc="Benchmarking"):
        B = 1  # Batch size 1 for now
        H = num_heads
        D = head_dim

        # Generate random Q, K
        q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = predictor(q, k)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Timing runs
        times = []
        for _ in range(num_trials):
            if device == 'cuda':
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                with torch.no_grad():
                    block_mask = predictor(q, k)
                end.record()

                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
                times.append(elapsed_ms)
            else:
                start = time.perf_counter()
                with torch.no_grad():
                    block_mask = predictor(q, k)
                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000
                times.append(elapsed_ms)

        # Statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)

        # FLOP estimate
        flop_info = predictor.estimate_flops(N, D)
        M = flop_info['M']

        # Memory usage
        if device == 'cuda':
            mem_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            torch.cuda.reset_peak_memory_stats()
        else:
            mem_allocated = 0

        results.append({
            'N': N,
            'M': M,
            'mean_ms': mean_time,
            'std_ms': std_time,
            'min_ms': min_time,
            'GFLOPS': flop_info['total'] / (mean_time * 1e6),  # GFLOP/s
            'mem_MB': mem_allocated,
        })

        print(f"\nN={N:>6,}  M={M:>4}  |  {mean_time:>6.2f} ± {std_time:>4.2f} ms  |  {mem_allocated:>6.1f} MB")

    df = pd.DataFrame(results)
    return df


def plot_benchmark_results(df: pd.DataFrame, save_path: str = 'predictor_benchmark.png'):
    """
    Visualizes benchmark results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Latency vs Sequence Length
    axes[0, 0].plot(df['N'], df['mean_ms'], marker='o', linewidth=2, color='#3498db')
    axes[0, 0].fill_between(df['N'], df['mean_ms'] - df['std_ms'], df['mean_ms'] + df['std_ms'],
                            alpha=0.3, color='#3498db')
    axes[0, 0].axhline(5.0, color='red', linestyle='--', linewidth=2, label='Target (5ms)')
    axes[0, 0].set_xlabel('Sequence Length (N)', fontsize=12)
    axes[0, 0].set_ylabel('Latency (ms)', fontsize=12)
    axes[0, 0].set_title('Predictor Latency vs Sequence Length', fontsize=14, fontweight='bold')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Memory Usage
    axes[0, 1].plot(df['N'], df['mem_MB'], marker='s', linewidth=2, color='#e74c3c')
    axes[0, 1].set_xlabel('Sequence Length (N)', fontsize=12)
    axes[0, 1].set_ylabel('Peak Memory (MB)', fontsize=12)
    axes[0, 1].set_title('Memory Usage', fontsize=14, fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # Throughput (GFLOP/s)
    axes[1, 0].plot(df['N'], df['GFLOPS'], marker='^', linewidth=2, color='#2ecc71')
    axes[1, 0].set_xlabel('Sequence Length (N)', fontsize=12)
    axes[1, 0].set_ylabel('Throughput (GFLOP/s)', fontsize=12)
    axes[1, 0].set_title('Compute Throughput', fontsize=14, fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Latency breakdown (log-log)
    axes[1, 1].loglog(df['N'], df['mean_ms'], marker='o', linewidth=2, color='#9b59b6', label='Total')
    # Theoretical O(M^2) line
    x = df['N'].values
    y_quad = df.loc[df['N'] == x[0], 'mean_ms'].values[0] * (x / x[0]) ** 2
    axes[1, 1].loglog(x, y_quad, linestyle='--', color='orange', linewidth=2, label='O(N²) Reference')
    # Actual scaling should be O(M^2) = O((N/64)^2) = O(N^2)
    axes[1, 1].set_xlabel('Sequence Length (N)', fontsize=12)
    axes[1, 1].set_ylabel('Latency (ms)', fontsize=12)
    axes[1, 1].set_title('Scaling Analysis (Log-Log)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, which='both')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.show()


def main():
    """
    Main benchmark routine.
    """
    # Test sequence lengths from 1k to 128k
    sequence_lengths = [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,  # 128k
    ]

    # Run benchmark
    df = benchmark_predictor(
        sequence_lengths=sequence_lengths,
        block_size=64,
        sparsity=0.95,
        num_heads=8,
        head_dim=128,
        num_trials=10,
        warmup=3,
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))

    # Check if we meet the 5ms target for N=128k
    if 131072 in df['N'].values:
        latency_128k = df.loc[df['N'] == 131072, 'mean_ms'].values[0]
        print("\n" + "=" * 80)
        if latency_128k < 5.0:
            print(f"✓ SUCCESS: Predictor latency at N=128k is {latency_128k:.2f}ms < 5ms target")
        else:
            print(f"✗ FAILURE: Predictor latency at N=128k is {latency_128k:.2f}ms > 5ms target")
        print("=" * 80)

    # Save results
    df.to_csv('predictor_benchmark.csv', index=False)
    print("\nSaved: predictor_benchmark.csv")

    # Plot
    plot_benchmark_results(df)


if __name__ == '__main__':
    main()
