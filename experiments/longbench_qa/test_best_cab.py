"""
Test Best CAB Configuration on Full Dataset

Based on parameter sweep results:
- CAB V3 (high FRC selection)
- Block size: 32
- Sparsity: 90%, 95%, 99%
"""

import sys
sys.path.insert(0, '../..')

import torch
import numpy as np
import json
from tqdm import tqdm
from typing import List
from transformers import GPT2Model, GPT2Tokenizer
from attention_preservation_test import load_longbench_narrativeqa


def apply_cab_v3_mask(
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 32,
    lambda_redundancy: float = 0.5
) -> torch.Tensor:
    """
    CAB V3: Select blocks with HIGHEST FRC.

    FRC = direct_attention - λ * redundancy
    High FRC = high direct attention but low redundancy
    = important unique connections
    """
    N = attention.shape[0]
    device = attention.device

    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=device)

    # Mean pooling
    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = attention[i_start:i_end, j_start:j_end].mean()

    # FRC
    redundancy = torch.matmul(block_scores, block_scores)
    frc_scores = block_scores - lambda_redundancy * redundancy

    k_keep = max(1, int(M * M * (1 - sparsity)))
    # KEY CHANGE: Select HIGHEST FRC (largest=True), not lowest
    threshold = torch.topk(frc_scores.flatten(), k_keep, largest=True).values[-1]
    block_mask = frc_scores >= threshold  # >= not <=

    token_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(M):
        for j in range(M):
            if block_mask[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask


def apply_h2o_mask(attention: torch.Tensor, sparsity: float, block_size: int = 32) -> torch.Tensor:
    """H2O baseline."""
    N = attention.shape[0]
    device = attention.device

    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=device)

    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = attention[i_start:i_end, j_start:j_end].max()

    k_keep = max(1, int(M * M * (1 - sparsity)))
    threshold = torch.topk(block_scores.flatten(), k_keep).values[-1]
    block_mask = block_scores >= threshold

    token_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(M):
        for j in range(M):
            if block_mask[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask


class BestCABAnalyzer:
    """Test best CAB configuration."""

    def __init__(self, device='cuda'):
        self.device = device
        print("Loading GPT-2...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2Model.from_pretrained(
            "gpt2",
            attn_implementation='eager'
        ).to(device)
        self.model.eval()
        print("✓ Loaded GPT-2")

    @torch.no_grad()
    def compute_atas(
        self,
        context: str,
        question: str,
        answer: str,
        method: str,
        sparsity: float,
        layer: int = 6
    ) -> float:
        """Compute ATAS."""
        full_text = f"{context} Question: {question}"
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(tokens) > 1024:
            tokens = tokens[:1024]

        N = len(tokens)

        # Find answer tokens
        answer_tokens_set = set(self.tokenizer.encode(answer, add_special_tokens=False))
        answer_positions = [i for i, token in enumerate(tokens) if token in answer_tokens_set]

        if not answer_positions:
            return 0.0

        # Extract attention
        input_ids = torch.tensor([tokens], device=self.device)
        outputs = self.model(input_ids, output_attentions=True, use_cache=False)
        attention = outputs.attentions[layer].mean(dim=1)[0]

        # Apply method (block_size=32 for all)
        if method == 'full':
            mask = torch.ones(N, N, dtype=torch.bool, device=self.device)
        elif method == 'h2o':
            mask = apply_h2o_mask(attention, sparsity, block_size=32)
        elif method == 'cab_v3':
            mask = apply_cab_v3_mask(attention, sparsity, block_size=32)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute ATAS
        question_start_pos = max(0, N - 50)
        total_attention = 0.0
        count = 0

        for query_pos in range(question_start_pos, N):
            for answer_pos in answer_positions:
                if mask[query_pos, answer_pos]:
                    total_attention += attention[query_pos, answer_pos].item()
                count += 1

        return total_attention / count if count > 0 else 0.0


def run_best_cab_test(num_samples: int = 20, device: str = 'cuda'):
    """Test best CAB configuration on full dataset."""

    print("=" * 80)
    print("BEST CAB CONFIGURATION TEST")
    print("=" * 80)
    print("Configuration from parameter sweep:")
    print("  - CAB V3 (select highest FRC)")
    print("  - Block size: 32")
    print("  - Sparsity: 90%, 95%, 99%")
    print("=" * 80)
    print()

    # Load dataset
    samples = load_longbench_narrativeqa(num_samples=num_samples)
    if samples is None or len(samples) == 0:
        print("ERROR: Could not load dataset")
        return

    print(f"Loaded {len(samples)} samples")

    # Initialize
    analyzer = BestCABAnalyzer(device=device)

    methods = ['full', 'h2o', 'cab_v3']
    sparsity_levels = [0.90, 0.95, 0.99]

    results = {m: {s: [] for s in sparsity_levels} for m in methods}

    total = len(samples) * len(methods) * len(sparsity_levels)
    pbar = tqdm(total=total, desc="Evaluating")

    for sample in samples:
        context = sample['context']
        question = sample['input']
        answer = sample['answers'][0] if isinstance(sample['answers'], list) else sample['answers']

        # Truncate
        context_words = context.split()
        if len(context_words) > 800:
            context = ' '.join(context_words[:800])

        for sparsity in sparsity_levels:
            for method in methods:
                atas = 0.0
                try:
                    atas = analyzer.compute_atas(
                        context, question, answer,
                        method=method,
                        sparsity=sparsity
                    )
                    results[method][sparsity].append(atas)

                except Exception as e:
                    results[method][sparsity].append(0.0)

                pbar.set_postfix({
                    'method': method,
                    'sparsity': f'{sparsity:.0%}',
                    'atas': f'{atas:.6f}'
                })
                pbar.update(1)

    pbar.close()

    # Summary
    summary = {}
    for method in methods:
        summary[method] = {}
        for sparsity in sparsity_levels:
            summary[method][sparsity] = {
                'mean': np.mean(results[method][sparsity]),
                'std': np.std(results[method][sparsity])
            }

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS: BEST CAB VS H2O (N=20)")
    print("=" * 80)

    for sparsity in sparsity_levels:
        print(f"\nSparsity: {sparsity:.0%}")
        print(f"{'Method':<10} | {'Mean ATAS':<12} | {'Std':<10} | {'vs Full':<10}")
        print("-" * 60)

        full_mean = summary['full'][sparsity]['mean']

        for method in methods:
            mean = summary[method][sparsity]['mean']
            std = summary[method][sparsity]['std']
            vs_full = (mean / full_mean * 100) if full_mean > 0 else 0

            marker = ""
            if method == 'cab_v3' and mean > summary['h2o'][sparsity]['mean']:
                marker = " 🏆"

            print(f"{method.upper():<10} | {mean:>11.6f} | {std:>9.6f} | {vs_full:>8.1f}%{marker}")

    # Save
    with open('best_cab_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✓ Saved results to best_cab_results.json")
    print("=" * 80)

    return summary


if __name__ == '__main__':
    results = run_best_cab_test(num_samples=20, device='cuda')
