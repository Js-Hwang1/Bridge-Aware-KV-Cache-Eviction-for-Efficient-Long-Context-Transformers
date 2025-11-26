"""
CAB Debugging and Parameter Sweep

This script:
1. Verifies FRC computation is correct
2. Sweeps λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
3. Sweeps block_size ∈ {32, 64, 128}
4. Tests alternative FRC formulations
5. Visualizes which blocks are selected
6. Diagnoses why CAB shows 0% preservation
"""

import sys
sys.path.insert(0, '../..')

import torch
import numpy as np
import json
import os
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import GPT2Model, GPT2Tokenizer


# ============================================================================
# Original CAB Implementation
# ============================================================================

def apply_cab_mask_original(
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 64,
    lambda_redundancy: float = 0.5
) -> torch.Tensor:
    """Original CAB: Lowest FRC (bridges)."""
    N = attention.shape[0]
    device = attention.device

    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=device)

    # Average pooling
    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = attention[i_start:i_end, j_start:j_end].mean()

    # FRC: direct - λ * redundancy
    redundancy = torch.matmul(block_scores, block_scores)
    frc_scores = block_scores - lambda_redundancy * redundancy

    k_keep = max(1, int(M * M * (1 - sparsity)))
    threshold = torch.topk(frc_scores.flatten(), k_keep, largest=False).values[-1]
    block_mask = frc_scores <= threshold

    token_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(M):
        for j in range(M):
            if block_mask[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask, block_scores, frc_scores, block_mask


# ============================================================================
# Alternative CAB Variants
# ============================================================================

def apply_cab_mask_v2(
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 64,
    lambda_redundancy: float = 0.5
) -> torch.Tensor:
    """
    CAB V2: Use max pooling instead of mean.

    Hypothesis: Mean pooling dilutes important connections.
    Max pooling preserves peak attention values.
    """
    N = attention.shape[0]
    device = attention.device

    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=device)

    # Max pooling (change from mean)
    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = attention[i_start:i_end, j_start:j_end].max()

    redundancy = torch.matmul(block_scores, block_scores)
    frc_scores = block_scores - lambda_redundancy * redundancy

    k_keep = max(1, int(M * M * (1 - sparsity)))
    threshold = torch.topk(frc_scores.flatten(), k_keep, largest=False).values[-1]
    block_mask = frc_scores <= threshold

    token_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(M):
        for j in range(M):
            if block_mask[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask, block_scores, frc_scores, block_mask


def apply_cab_mask_v3(
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 64,
    lambda_redundancy: float = 0.5
) -> torch.Tensor:
    """
    CAB V3: Select HIGH FRC instead of low.

    Hypothesis: Maybe we got the logic backwards?
    High FRC = high direct attention - low redundancy = important unique connections
    """
    N = attention.shape[0]
    device = attention.device

    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=device)

    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = attention[i_start:i_end, j_start:j_end].mean()

    redundancy = torch.matmul(block_scores, block_scores)
    frc_scores = block_scores - lambda_redundancy * redundancy

    k_keep = max(1, int(M * M * (1 - sparsity)))
    # Change: select HIGHEST FRC (largest=True)
    threshold = torch.topk(frc_scores.flatten(), k_keep, largest=True).values[-1]
    block_mask = frc_scores >= threshold  # Change: >= instead of <=

    token_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(M):
        for j in range(M):
            if block_mask[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask, block_scores, frc_scores, block_mask


def apply_cab_mask_v4(
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 64,
    lambda_redundancy: float = 0.5
) -> torch.Tensor:
    """
    CAB V4: Hybrid - combine magnitude and FRC.

    score = α × magnitude + (1-α) × (-FRC)
    where α controls trade-off
    """
    N = attention.shape[0]
    device = attention.device

    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=device)

    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = attention[i_start:i_end, j_start:j_end].mean()

    # Compute FRC
    redundancy = torch.matmul(block_scores, block_scores)
    frc_scores = block_scores - lambda_redundancy * redundancy

    # Hybrid: 50% magnitude, 50% inverted FRC
    alpha = 0.5
    hybrid_scores = alpha * block_scores + (1 - alpha) * (-frc_scores)

    k_keep = max(1, int(M * M * (1 - sparsity)))
    threshold = torch.topk(hybrid_scores.flatten(), k_keep, largest=True).values[-1]
    block_mask = hybrid_scores >= threshold

    token_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(M):
        for j in range(M):
            if block_mask[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask, block_scores, frc_scores, block_mask


def apply_h2o_mask(attention: torch.Tensor, sparsity: float, block_size: int = 64) -> Tuple:
    """H2O for comparison."""
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

    return token_mask, block_scores, None, block_mask


# ============================================================================
# Diagnostic Analyzer
# ============================================================================

class DiagnosticAnalyzer:
    """Analyzes CAB behavior with detailed diagnostics."""

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
    def analyze_sample(
        self,
        context: str,
        question: str,
        answer: str,
        method: str,
        sparsity: float,
        block_size: int,
        lambda_redundancy: float,
        layer: int = 6
    ) -> Dict:
        """Analyze a single sample with full diagnostics."""

        # Construct input
        full_text = f"{context} Question: {question}"
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(tokens) > 1024:
            tokens = tokens[:1024]

        N = len(tokens)

        # Find answer positions
        answer_tokens_set = set(self.tokenizer.encode(answer, add_special_tokens=False))
        answer_positions = [i for i, token in enumerate(tokens) if token in answer_tokens_set]

        if not answer_positions:
            return {'error': 'Answer not found in context'}

        # Extract attention
        input_ids = torch.tensor([tokens], device=self.device)
        outputs = self.model(input_ids, output_attentions=True, use_cache=False)
        attention = outputs.attentions[layer].mean(dim=1)[0]  # [N, N]

        # Apply method
        if method == 'h2o':
            mask, block_scores, frc_scores, block_mask = apply_h2o_mask(attention, sparsity, block_size)
        elif method == 'cab_original':
            mask, block_scores, frc_scores, block_mask = apply_cab_mask_original(attention, sparsity, block_size, lambda_redundancy)
        elif method == 'cab_v2':
            mask, block_scores, frc_scores, block_mask = apply_cab_mask_v2(attention, sparsity, block_size, lambda_redundancy)
        elif method == 'cab_v3':
            mask, block_scores, frc_scores, block_mask = apply_cab_mask_v3(attention, sparsity, block_size, lambda_redundancy)
        elif method == 'cab_v4':
            mask, block_scores, frc_scores, block_mask = apply_cab_mask_v4(attention, sparsity, block_size, lambda_redundancy)
        else:
            mask = torch.ones(N, N, dtype=torch.bool, device=self.device)
            block_scores = None
            frc_scores = None
            block_mask = None

        # Compute ATAS
        question_start_pos = max(0, N - 50)
        total_attention = 0.0
        count = 0

        for query_pos in range(question_start_pos, N):
            for answer_pos in answer_positions:
                if mask[query_pos, answer_pos]:
                    total_attention += attention[query_pos, answer_pos].item()
                count += 1

        atas = total_attention / count if count > 0 else 0.0

        # Diagnostics: which blocks contain answer tokens?
        M = (N + block_size - 1) // block_size
        answer_blocks = set()
        for pos in answer_positions:
            block_idx = pos // block_size
            answer_blocks.add(block_idx)

        # How many answer-containing blocks were selected?
        if block_mask is not None:
            answer_blocks_selected = 0
            for block_i in answer_blocks:
                # Check if ANY block in this column is selected (column = key position)
                if block_mask[:, block_i].any():
                    answer_blocks_selected += 1

            answer_block_coverage = answer_blocks_selected / len(answer_blocks) if answer_blocks else 0.0
        else:
            answer_block_coverage = 1.0  # Full attention

        return {
            'atas': atas,
            'answer_positions': answer_positions,
            'answer_blocks': list(answer_blocks),
            'answer_block_coverage': answer_block_coverage,
            'num_blocks_total': M * M,
            'num_blocks_kept': int(block_mask.sum().item()) if block_mask is not None else M * M,
            'block_scores_stats': {
                'min': block_scores.min().item() if block_scores is not None else 0,
                'max': block_scores.max().item() if block_scores is not None else 0,
                'mean': block_scores.mean().item() if block_scores is not None else 0,
            } if block_scores is not None else None,
            'frc_scores_stats': {
                'min': frc_scores.min().item() if frc_scores is not None else 0,
                'max': frc_scores.max().item() if frc_scores is not None else 0,
                'mean': frc_scores.mean().item() if frc_scores is not None else 0,
            } if frc_scores is not None else None,
        }


# ============================================================================
# Parameter Sweep
# ============================================================================

def run_parameter_sweep(num_samples: int = 10, device: str = 'cuda'):
    """Run comprehensive parameter sweep."""

    print("=" * 80)
    print("CAB PARAMETER SWEEP & DEBUGGING")
    print("=" * 80)
    print()

    # Load small dataset
    from attention_preservation_test import load_longbench_narrativeqa
    samples = load_longbench_narrativeqa(num_samples=num_samples)

    if samples is None or len(samples) == 0:
        print("ERROR: Could not load dataset")
        return

    print(f"Loaded {len(samples)} samples")

    # Initialize analyzer
    analyzer = DiagnosticAnalyzer(device=device)

    # Parameter grid
    methods = ['h2o', 'cab_original', 'cab_v2', 'cab_v3', 'cab_v4']
    sparsity_levels = [0.90, 0.95]
    block_sizes = [32, 64, 128]
    lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    results = {}

    total = len(samples) * len(methods) * len(sparsity_levels) * len(block_sizes) * len(lambda_values)
    pbar = tqdm(total=total, desc="Sweeping")

    for sample_idx, sample in enumerate(samples):
        context = sample['context']
        question = sample['input']
        answer = sample['answers'][0] if isinstance(sample['answers'], list) else sample['answers']

        # Truncate context
        context_words = context.split()
        if len(context_words) > 800:
            context = ' '.join(context_words[:800])

        for method in methods:
            for sparsity in sparsity_levels:
                for block_size in block_sizes:
                    for lambda_val in lambda_values:
                        # Skip lambda for non-CAB methods
                        if 'cab' not in method and lambda_val != 0.5:
                            pbar.update(1)
                            continue

                        key = (method, sparsity, block_size, lambda_val)

                        if key not in results:
                            results[key] = {
                                'atas_scores': [],
                                'answer_coverages': [],
                                'diagnostics': []
                            }

                        try:
                            diag = analyzer.analyze_sample(
                                context, question, answer,
                                method=method,
                                sparsity=sparsity,
                                block_size=block_size,
                                lambda_redundancy=lambda_val
                            )

                            if 'error' not in diag:
                                results[key]['atas_scores'].append(diag['atas'])
                                results[key]['answer_coverages'].append(diag['answer_block_coverage'])
                                if sample_idx == 0:  # Store detailed diagnostics for first sample
                                    results[key]['diagnostics'].append(diag)

                        except Exception as e:
                            pass

                        pbar.update(1)

    pbar.close()

    # Summarize results
    summary = {}
    for key, data in results.items():
        method, sparsity, block_size, lambda_val = key

        if len(data['atas_scores']) > 0:
            summary[key] = {
                'method': method,
                'sparsity': sparsity,
                'block_size': block_size,
                'lambda': lambda_val,
                'mean_atas': np.mean(data['atas_scores']),
                'std_atas': np.std(data['atas_scores']),
                'mean_answer_coverage': np.mean(data['answer_coverages']),
                'diagnostics': data['diagnostics'][:1]  # First sample only
            }

    # Sort by mean_atas
    sorted_results = sorted(summary.items(), key=lambda x: x[1]['mean_atas'], reverse=True)

    # Print top 20 configurations
    print("\n" + "=" * 80)
    print("TOP 20 CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Method':<15} | {'Sparsity':<8} | {'Block':<5} | {'Lambda':<6} | {'ATAS':<10} | {'Coverage':<10}")
    print("-" * 90)

    for i, (key, result) in enumerate(sorted_results[:20]):
        print(f"{result['method']:<15} | {result['sparsity']:<8.0%} | {result['block_size']:<5} | {result['lambda']:<6.2f} | {result['mean_atas']:<10.6f} | {result['mean_answer_coverage']:<10.2%}")

    # Save results
    with open('cab_parameter_sweep_results.json', 'w') as f:
        # Convert keys to strings for JSON
        json_results = {str(k): v for k, v in summary.items()}
        json.dump(json_results, f, indent=2)

    print("\n✓ Saved results to cab_parameter_sweep_results.json")
    print("=" * 80)

    return summary


if __name__ == '__main__':
    results = run_parameter_sweep(num_samples=10, device='cuda')
