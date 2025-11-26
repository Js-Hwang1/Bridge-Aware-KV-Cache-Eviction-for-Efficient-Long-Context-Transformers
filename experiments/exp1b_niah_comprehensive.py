#!/usr/bin/env python3
"""
Comprehensive NIAH Benchmark for ICML Submission
TODO 1.1: Full NIAH Suite

Tests:
- Multi-needle retrieval (1, 2, 3, 5 needles)
- Extended context lengths (2K, 4K, 8K, 16K, 32K)
- Multiple sparsity levels (85%, 90%, 95%, 99%)
- Variable needle positions across context
- CAB V4 hybrid variants vs H2O baseline
"""

import torch
import torch.nn.functional as F
import json
import time
import sys
import os
from typing import List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cab_attention.kernels.coarsening import coarsen_qk_max_l2
from cab_attention.kernels.frc_kernel import compute_block_frc, generate_block_mask


def generate_multi_needle_haystack(
    seq_len: int,
    num_needles: int,
    needle_depths: List[float],
    vocab_size: int = 50000,
    d_model: int = 64,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Generate haystack with multiple needles at specified depths.

    Returns:
        q, k tensors and list of needle positions
    """
    # Generate random haystack
    q = torch.randn(1, 1, seq_len, d_model, device=device) * 0.1
    k = torch.randn(1, 1, seq_len, d_model, device=device) * 0.1

    # Place needles at specified depths
    needle_positions = []
    for i, depth in enumerate(needle_depths[:num_needles]):
        pos = int(depth * (seq_len - 1))
        # Make needle distinctive with high magnitude
        q[0, 0, pos] = torch.randn(d_model, device=device) * 2.0  # Query for needle
        k[0, 0, pos] = q[0, 0, pos] + torch.randn(d_model, device=device) * 0.1  # Similar key
        needle_positions.append(pos)

    return q, k, needle_positions


def apply_h2o(
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 32,
) -> Tuple[torch.Tensor, float]:
    """Apply H2O (Heavy-Hitter Oracle) sparse attention."""
    start = time.time()

    B, H, N, _ = attention.shape

    # Compute magnitude scores (max attention per position)
    magnitude_scores = attention.max(dim=2)[0]  # [B, H, N]

    # Select top-k positions
    k = int(N * (1 - sparsity))
    topk_indices = magnitude_scores.topk(k, dim=-1).indices

    # Create mask
    mask = torch.zeros_like(attention)
    for b in range(B):
        for h in range(H):
            mask[b, h, :, topk_indices[b, h]] = 1.0

    # Apply mask
    sparse_attention = attention * mask
    row_sums = sparse_attention.sum(dim=-1, keepdim=True) + 1e-8
    sparse_attention = sparse_attention / row_sums

    elapsed = (time.time() - start) * 1000
    return sparse_attention, elapsed


def apply_cab_v4(
    q: torch.Tensor,
    k: torch.Tensor,
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 32,
    magnitude_ratio: float = 0.5,
) -> Tuple[torch.Tensor, float, float]:
    """Apply CAB V4 Hybrid sparse attention."""
    start = time.time()

    B, H, N, D = q.shape
    M = (N + block_size - 1) // block_size

    # Coarsen Q/K using optimized Triton kernel
    q_coarse, k_coarse = coarsen_qk_max_l2(q, k, block_size=block_size)

    # Compute FRC scores
    frc_scores, affinity, redundancy = compute_block_frc(
        q_coarse, k_coarse,
        formula='additive',
        normalization='minmax',
        lambda_redundancy=0.3,
    )

    # Compute magnitude scores (H2O style at block level)
    attention_blocks = attention.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    magnitude_scores = attention_blocks.max(dim=-1)[0].max(dim=-1)[0]  # [B, H, M, M]

    # Generate hybrid block mask
    block_mask = generate_block_mask(
        frc_scores,
        sparsity=sparsity,
        magnitude_scores=magnitude_scores,
        magnitude_ratio=magnitude_ratio,
        select_high=True,
        keep_diagonal=True,
    )

    # Expand block mask to token level
    token_mask = block_mask.float().repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
    token_mask = token_mask[..., :N, :N]

    # Apply mask
    sparse_attention = attention * token_mask
    row_sums = sparse_attention.sum(dim=-1, keepdim=True) + 1e-8
    sparse_attention = sparse_attention / row_sums

    elapsed = (time.time() - start) * 1000

    # Compute discriminative power
    disc_power = (frc_scores.max() - frc_scores.min()).item()

    return sparse_attention, elapsed, disc_power


def evaluate_needle_retrieval(
    sparse_attention: torch.Tensor,
    needle_positions: List[int],
    threshold: float = 0.01,
) -> float:
    """
    Evaluate if needles are correctly retrieved.

    Returns fraction of needles successfully retrieved (0.0 to 1.0).
    """
    # Check if attention focuses on needle positions
    # For each needle, check if it gets high attention from query token 0
    retrieved = 0

    for pos in needle_positions:
        # Check if needle position gets attention above threshold
        attn_to_needle = sparse_attention[0, 0, 0, pos].item()
        if attn_to_needle > threshold:
            retrieved += 1

    return retrieved / len(needle_positions) if needle_positions else 0.0


def run_niah_benchmark():
    """Run comprehensive NIAH benchmark."""

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires GPU.")
        return

    device = 'cuda'
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Configuration
    config = {
        'context_lengths': [2048, 4096, 8192, 16384, 32768],
        'sparsity_levels': [0.85, 0.90, 0.95, 0.99],
        'needle_counts': [1, 2, 3, 5],
        'needle_depths': [0.0, 0.25, 0.5, 0.75, 1.0],
        'magnitude_ratios': [0.0, 0.25, 0.5, 0.75, 1.0],
        'trials_per_config': 3,
        'block_size': 32,
        'B': 1,
        'H': 8,
        'D': 64,
    }

    results = {
        'config': config,
        'data': []
    }

    # Calculate total experiments
    total_configs = (
        len(config['context_lengths']) *
        len(config['sparsity_levels']) *
        len(config['needle_counts']) *
        len(config['needle_depths']) *
        config['trials_per_config']
    )

    print(f"Starting Comprehensive NIAH Benchmark")
    print(f"Total configurations: {total_configs}")
    print(f"Methods: Dense, H2O, CAB(0.0), CAB(0.25), CAB(0.5), CAB(0.75), CAB(1.0)")
    print(f"Estimated time: 2-4 hours")
    print("=" * 80)

    experiment_num = 0

    for context_len in config['context_lengths']:
        for sparsity in config['sparsity_levels']:
            for num_needles in config['needle_counts']:
                for needle_depth in config['needle_depths']:

                    # Generate needle depth distribution
                    # For multi-needle, distribute across context
                    if num_needles == 1:
                        depths = [needle_depth]
                    else:
                        # Distribute needles around the specified depth
                        spread = 0.2 / num_needles
                        depths = [max(0.0, min(1.0, needle_depth + i * spread - spread * (num_needles-1)/2))
                                  for i in range(num_needles)]

                    print(f"\n{'=' * 80}")
                    print(f"N={context_len}, Sparsity={int(sparsity*100)}%, "
                          f"Needles={num_needles}, Depth={int(needle_depth*100)}%")
                    print(f"{'=' * 80}\n")

                    trials_data = []

                    for trial in range(config['trials_per_config']):
                        experiment_num += 1
                        print(f"[{experiment_num}/{total_configs}] Trial {trial+1}/{config['trials_per_config']}")

                        # Generate multi-needle problem
                        q, k, needle_positions = generate_multi_needle_haystack(
                            seq_len=context_len,
                            num_needles=num_needles,
                            needle_depths=depths,
                            d_model=config['D'],
                            device=device,
                        )

                        # Compute dense attention (oracle)
                        scores = torch.matmul(q, k.transpose(-2, -1)) / (config['D'] ** 0.5)
                        attention = F.softmax(scores, dim=-1)

                        dense_recall = evaluate_needle_retrieval(attention, needle_positions)
                        print(f"  Dense: {dense_recall:.3f}")

                        # H2O baseline
                        h2o_attn, h2o_time = apply_h2o(attention, sparsity, config['block_size'])
                        h2o_recall = evaluate_needle_retrieval(h2o_attn, needle_positions)
                        print(f"  H2O:   {h2o_recall:.3f} ({h2o_time:.2f}ms)")

                        # CAB V4 variants
                        cab_results = {}
                        for mag_ratio in config['magnitude_ratios']:
                            cab_attn, cab_time, disc_power = apply_cab_v4(
                                q, k, attention, sparsity,
                                block_size=config['block_size'],
                                magnitude_ratio=mag_ratio,
                            )
                            cab_recall = evaluate_needle_retrieval(cab_attn, needle_positions)

                            cab_results[f'mag_{mag_ratio:.2f}'] = {
                                'recall': cab_recall,
                                'time_ms': cab_time,
                                'disc_power': disc_power,
                            }

                            print(f"  CAB({mag_ratio:.2f}): {cab_recall:.3f} ({cab_time:.2f}ms)")

                        # Store trial data
                        trials_data.append({
                            'needle_positions': needle_positions,
                            'needle_depths': depths,
                            'dense_recall': dense_recall,
                            'h2o': {
                                'recall': h2o_recall,
                                'time_ms': h2o_time,
                            },
                            'cab_variants': cab_results,
                        })

                    # Store configuration results
                    results['data'].append({
                        'context_length': context_len,
                        'sparsity': sparsity,
                        'num_needles': num_needles,
                        'needle_depth': needle_depth,
                        'trials': trials_data,
                    })

                    # Save intermediate results
                    with open('results/exp1b_niah_comprehensive.json', 'w') as f:
                        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Comprehensive NIAH Benchmark Complete!")
    print(f"Results saved to: results/exp1b_niah_comprehensive.json")
    print("=" * 80)


if __name__ == '__main__':
    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Run benchmark
    run_niah_benchmark()
