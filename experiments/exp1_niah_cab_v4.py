"""
Experiment 1: NIAH Benchmark - CAB V4 vs H2O

This is the main experiment for ICML paper data collection.

Tests:
- H2O (baseline)
- CAB V4 Hybrid (50/50 magnitude + FRC)
- CAB V4 variants (25/75, 75/25)

Metrics:
- Needle recall at various depths
- Performance across sparsity levels (85%, 90%, 95%)
- Computational efficiency
- Discriminative power

Output: results/exp1_niah_results.json (paper-ready data)
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cab_attention.kernels.frc_kernel import compute_block_frc, generate_block_mask
from cab_attention.kernels.coarsening import coarsen_qk_max_l2


class NIAHDataset:
    """Generate NIAH samples with realistic token distributions."""
    def __init__(self, vocab_size=50257, device='cuda'):
        self.vocab_size = vocab_size
        self.device = device

        # Zipfian distribution for realistic tokens
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32)
        probs = 1.0 / ranks ** 1.1
        self.token_probs = (probs / probs.sum()).to(device)

    def generate_sample(self, context_length: int, needle_depth: float):
        """Generate one NIAH sample."""
        # Background tokens
        input_ids = torch.multinomial(
            self.token_probs,
            num_samples=context_length,
            replacement=True
        )

        # Needle token (rare token)
        needle_token = torch.randint(
            self.vocab_size - 100,
            self.vocab_size,
            (1,),
            device=self.device
        ).item()

        # Insert needle
        needle_pos = max(1, min(int(context_length * needle_depth), context_length - 2))
        input_ids[needle_pos] = needle_token

        return {
            'input_ids': input_ids,
            'needle_position': needle_pos,
            'needle_token': needle_token,
        }


def compute_needle_recall(attention: torch.Tensor, needle_pos: int, top_k: int = 10) -> float:
    """
    Compute needle recall: is needle in top-k attended positions?

    Averages over queries near the needle (±10 tokens).
    """
    avg_attention = attention.mean(dim=(0, 1))  # [N, N]

    recalls = []
    query_range = range(max(0, needle_pos - 10), min(avg_attention.shape[0], needle_pos + 10))

    for q_pos in query_range:
        top_k_indices = torch.topk(avg_attention[q_pos], k=top_k).indices
        is_recalled = needle_pos in top_k_indices.tolist()
        recalls.append(float(is_recalled))

    return np.mean(recalls) if recalls else 0.0


def apply_h2o(attention: torch.Tensor, sparsity: float, block_size: int = 32):
    """Apply H2O sparse attention (magnitude-based)."""
    start = time.time()

    N = attention.shape[-1]
    M = (N + block_size - 1) // block_size

    # Max pooling to get block scores
    attention_blocks = attention.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    block_scores = attention_blocks.max(dim=-1)[0].max(dim=-1)[0]  # [B, H, M, M]

    # Select top-k blocks
    k_keep = max(1, int(M * M * (1.0 - sparsity)))
    threshold = torch.topk(block_scores.flatten(2), k_keep, dim=-1).values[:, :, -1:]
    threshold = threshold.unsqueeze(-1)

    # Create mask
    block_mask = (block_scores >= threshold).float()
    token_mask = block_mask.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
    token_mask = token_mask[..., :N, :N]

    # Apply and renormalize
    sparse_attention = attention * token_mask
    row_sums = sparse_attention.sum(dim=-1, keepdim=True) + 1e-8
    sparse_attention = sparse_attention / row_sums

    elapsed = (time.time() - start) * 1000

    return sparse_attention, elapsed, block_scores


def apply_cab_v4(
    q: torch.Tensor,
    k: torch.Tensor,
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 32,
    magnitude_ratio: float = 0.5,
):
    """Apply CAB V4 Hybrid sparse attention."""
    start = time.time()

    B, H, N, D = q.shape
    M = (N + block_size - 1) // block_size

    # Coarsen Q/K
    q_coarse, k_coarse = coarsen_qk_max_l2(q, k, block_size=block_size)

    # Compute FRC
    frc_scores, affinity, redundancy = compute_block_frc(
        q_coarse, k_coarse,
        formula='additive',
        normalization='minmax',
        lambda_redundancy=0.3,
    )

    # Compute magnitude scores (H2O style)
    attention_blocks = attention.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    magnitude_scores = attention_blocks.max(dim=-1)[0].max(dim=-1)[0]

    # Generate hybrid mask
    block_mask = generate_block_mask(
        frc_scores,
        sparsity=sparsity,
        magnitude_scores=magnitude_scores,
        magnitude_ratio=magnitude_ratio,
        select_high=True,
        keep_diagonal=True,
    )

    # Expand to tokens
    token_mask = block_mask.float().repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
    token_mask = token_mask[..., :N, :N]

    # Apply and renormalize
    sparse_attention = attention * token_mask
    row_sums = sparse_attention.sum(dim=-1, keepdim=True) + 1e-8
    sparse_attention = sparse_attention / row_sums

    elapsed = (time.time() - start) * 1000

    # Diagnostics
    from cab_attention.kernels.frc_kernel import analyze_frc_discriminative_power
    disc_power = analyze_frc_discriminative_power(frc_scores, sparsity=sparsity, verbose=False)

    return sparse_attention, elapsed, {
        'frc_scores': frc_scores,
        'magnitude_scores': magnitude_scores,
        'discriminative_power': disc_power['discriminative_power'],
        'magnitude_ratio': magnitude_ratio,
    }


def run_experiment():
    """Run main NIAH experiment."""
    print("="*100)
    print("EXPERIMENT 1: NIAH BENCHMARK - CAB V4 vs H2O")
    print("="*100)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cpu':
        print("WARNING: Running on CPU will be very slow!")

    # Configuration
    config = {
        'context_lengths': [1024, 2048, 4096],
        'sparsity_levels': [0.85, 0.90, 0.95],
        'needle_depths': [0.0, 0.25, 0.5, 0.75, 1.0],
        'magnitude_ratios': [0.0, 0.25, 0.5, 0.75, 1.0],  # 0=pure FRC, 1=pure H2O
        'trials_per_config': 5,
        'block_size': 32,
        'B': 1,  # batch
        'H': 8,  # heads
        'D': 64,  # dim
    }

    print(f"\nConfiguration:")
    print(f"  Context lengths: {config['context_lengths']}")
    print(f"  Sparsity levels: {[f'{s*100:.0f}%' for s in config['sparsity_levels']]}")
    print(f"  Needle depths: {[f'{d*100:.0f}%' for d in config['needle_depths']]}")
    print(f"  CAB variants: magnitude_ratio={config['magnitude_ratios']}")
    print(f"  Trials per config: {config['trials_per_config']}")
    print()

    dataset = NIAHDataset(device=device)
    results = {'config': config, 'data': []}

    total_tests = (
        len(config['context_lengths']) *
        len(config['sparsity_levels']) *
        len(config['needle_depths']) *
        config['trials_per_config']
    )

    test_num = 0

    for N in config['context_lengths']:
        for sparsity in config['sparsity_levels']:
            for needle_depth in config['needle_depths']:
                print(f"\n{'='*80}")
                print(f"N={N}, Sparsity={sparsity*100:.0f}%, Depth={needle_depth*100:.0f}%")
                print(f"{'='*80}")

                trial_results = {
                    'context_length': N,
                    'sparsity': sparsity,
                    'needle_depth': needle_depth,
                    'trials': [],
                }

                for trial in range(config['trials_per_config']):
                    test_num += 1
                    print(f"\n[{test_num}/{total_tests}] Trial {trial+1}/{config['trials_per_config']}")

                    # Generate sample
                    sample = dataset.generate_sample(N, needle_depth)
                    needle_pos = sample['needle_position']

                    # Create random Q, K (simulating embeddings)
                    q = torch.randn(config['B'], config['H'], N, config['D'], device=device)
                    k = torch.randn(config['B'], config['H'], N, config['D'], device=device)

                    # Boost needle signal
                    for i in range(max(0, needle_pos - 20), min(N, needle_pos + 20)):
                        q[:, :, i, :] = k[:, :, needle_pos, :] + torch.randn(
                            config['B'], config['H'], config['D'], device=device
                        ) * 0.1

                    # Compute full attention
                    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (config['D'] ** 0.5)
                    attention_full = F.softmax(attention_scores, dim=-1)

                    # Dense recall
                    dense_recall = compute_needle_recall(attention_full, needle_pos)

                    # H2O
                    attn_h2o, time_h2o, _ = apply_h2o(
                        attention_full, sparsity, config['block_size']
                    )
                    h2o_recall = compute_needle_recall(attn_h2o, needle_pos)

                    # CAB V4 variants
                    cab_results = {}
                    for mag_ratio in config['magnitude_ratios']:
                        attn_cab, time_cab, diag = apply_cab_v4(
                            q, k, attention_full, sparsity,
                            block_size=config['block_size'],
                            magnitude_ratio=mag_ratio,
                        )
                        cab_recall = compute_needle_recall(attn_cab, needle_pos)

                        cab_results[f'mag_{mag_ratio:.2f}'] = {
                            'recall': cab_recall,
                            'time_ms': time_cab,
                            'disc_power': diag['discriminative_power'],
                        }

                    trial_data = {
                        'needle_position': needle_pos,
                        'dense_recall': dense_recall,
                        'h2o': {
                            'recall': h2o_recall,
                            'time_ms': time_h2o,
                        },
                        'cab_variants': cab_results,
                    }

                    trial_results['trials'].append(trial_data)

                    # Print summary
                    print(f"  Dense: {dense_recall:.3f}")
                    print(f"  H2O:   {h2o_recall:.3f} ({time_h2o:.2f}ms)")
                    for mag_ratio in config['magnitude_ratios']:
                        cr = cab_results[f'mag_{mag_ratio:.2f}']
                        print(f"  CAB({mag_ratio:.2f}): {cr['recall']:.3f} ({cr['time_ms']:.2f}ms)")

                results['data'].append(trial_results)

    # Save results
    output_path = Path(__file__).parent / 'results' / 'exp1_niah_results.json'
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*100}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*100}")

    # Quick summary
    print("\n=== QUICK SUMMARY ===\n")

    for sparsity in config['sparsity_levels']:
        sparsity_data = [d for d in results['data'] if d['sparsity'] == sparsity]

        h2o_recalls = []
        cab_50_recalls = []  # 50/50 hybrid

        for entry in sparsity_data:
            for trial in entry['trials']:
                h2o_recalls.append(trial['h2o']['recall'])
                cab_50_recalls.append(trial['cab_variants']['mag_0.50']['recall'])

        print(f"Sparsity {sparsity*100:.0f}%:")
        print(f"  H2O:        {np.mean(h2o_recalls):.3f} ± {np.std(h2o_recalls):.3f}")
        print(f"  CAB V4(50): {np.mean(cab_50_recalls):.3f} ± {np.std(cab_50_recalls):.3f}")

        if np.mean(cab_50_recalls) >= np.mean(h2o_recalls) * 0.95:
            print(f"  → CAB V4 matches H2O ✓")
        else:
            print(f"  → CAB V4 needs improvement")
        print()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    run_experiment()
