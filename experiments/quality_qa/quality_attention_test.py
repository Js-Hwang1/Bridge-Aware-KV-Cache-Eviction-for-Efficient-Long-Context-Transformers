"""
QuALITY Attention Preservation Test

QuALITY: Question Answering with Long Input Texts, Yes!
- Multiple-choice questions requiring reasoning over long articles
- Questions designed to be difficult without reading the full document
- Better for attention analysis than abstractive QA
"""

import sys
sys.path.insert(0, '../..')

import torch
import numpy as np
import json
import os
from tqdm import tqdm
from typing import List, Dict
from transformers import GPT2Model, GPT2Tokenizer
from urllib.request import urlopen
from urllib.error import URLError


# ============================================================================
# Sparse Attention Masks
# ============================================================================

def apply_h2o_mask(attention: torch.Tensor, sparsity: float, block_size: int = 64) -> torch.Tensor:
    """H2O: Top-k blocks by magnitude."""
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


def apply_cab_mask(attention: torch.Tensor, sparsity: float, block_size: int = 64, lambda_redundancy: float = 0.5) -> torch.Tensor:
    """CAB: Lowest FRC (bridges)."""
    N = attention.shape[0]
    device = attention.device

    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=device)

    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = attention[i_start:i_end, j_start:j_end].mean()

    # FRC
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

    return token_mask


# ============================================================================
# Dataset Loading
# ============================================================================

def load_quality_dataset(num_samples: int = 20, cache_dir: str = './data'):
    """Load QuALITY dataset from HuggingFace."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'quality.jsonl')

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        print("Loading QuALITY from HuggingFace...")
        dataset = load_dataset('quality', split='validation')  # Use validation split

        samples = []
        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            # QuALITY format: article, question, options (list of 4), correct_option (0-3)
            samples.append({
                'context': item['article'],
                'question': item['question'],
                'options': item['options'],
                'answer_idx': item['correct_option'],
                'answer': item['options'][item['correct_option']]
            })

        if len(samples) > 0:
            print(f"✓ Loaded {len(samples)} samples from QuALITY")
            with open(cache_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            return samples

    except Exception as e:
        print(f"QuALITY loading failed: {e}")

    # Use cached file
    if os.path.exists(cache_file):
        print(f"Using cached dataset: {cache_file}")
        samples = []
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
                    if len(samples) >= num_samples:
                        break
        if len(samples) > 0:
            print(f"✓ Loaded {len(samples)} samples from cache")
            return samples

    print("Could not load QuALITY dataset")
    return None


# ============================================================================
# Attention Analyzer
# ============================================================================

class QuALITYAttentionAnalyzer:
    """Analyzes attention preservation for QuALITY multiple-choice QA."""

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
    def compute_option_atas(
        self,
        context: str,
        question: str,
        correct_option: str,
        method: str = 'full',
        sparsity: float = 0.95,
        layer: int = 6
    ) -> float:
        """
        Compute ATAS for the correct answer option.

        For multiple-choice, we measure attention to tokens in the correct option.
        """
        # Construct input
        full_text = f"{context} Question: {question}"

        # Tokenize
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate if needed
        if len(tokens) > 1024:
            tokens = tokens[:1024]

        N = len(tokens)

        # Find correct option tokens
        option_tokens_set = set(self.tokenizer.encode(correct_option, add_special_tokens=False))

        option_positions = []
        for i, token in enumerate(tokens):
            if token in option_tokens_set:
                option_positions.append(i)

        if not option_positions:
            # Option text not found in context
            return 0.0

        # Extract attention
        input_ids = torch.tensor([tokens], device=self.device)
        outputs = self.model(input_ids, output_attentions=True, use_cache=False)
        attention = outputs.attentions[layer].mean(dim=1)[0]  # [N, N]

        # Apply sparsity
        if method == 'full':
            mask = torch.ones(N, N, dtype=torch.bool, device=self.device)
        elif method == 'h2o':
            mask = apply_h2o_mask(attention, sparsity)
        elif method == 'cab':
            mask = apply_cab_mask(attention, sparsity)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute ATAS: attention from query tokens to option tokens
        question_start_pos = max(0, N - 50)

        total_attention = 0.0
        count = 0

        for query_pos in range(question_start_pos, N):
            for option_pos in option_positions:
                if mask[query_pos, option_pos]:
                    total_attention += attention[query_pos, option_pos].item()
                count += 1

        return total_attention / count if count > 0 else 0.0


# ============================================================================
# Experiment Runner
# ============================================================================

def run_quality_test(
    num_samples: int = 20,
    methods: List[str] = ['full', 'h2o', 'cab'],
    sparsity_levels: List[float] = [0.90, 0.95, 0.99],
    device: str = 'cuda'
):
    """Run QuALITY attention preservation test."""

    print("=" * 80)
    print("QuALITY ATTENTION PRESERVATION TEST")
    print("=" * 80)
    print("Dataset: QuALITY (Multiple-Choice QA)")
    print("Metric: Answer Token Attention Score (ATAS)")
    print("=" * 80)
    print(f"Methods: {methods}")
    print(f"Sparsity: {sparsity_levels}")
    print(f"Samples: {num_samples}")
    print("=" * 80)
    print()

    # Load dataset
    samples = load_quality_dataset(num_samples=num_samples)

    if samples is None or len(samples) == 0:
        print("ERROR: Could not load QuALITY dataset")
        return None

    # Initialize analyzer
    analyzer = QuALITYAttentionAnalyzer(device=device)

    # Results
    results = {m: {s: [] for s in sparsity_levels} for m in methods}

    # Evaluate
    total = len(samples) * len(methods) * len(sparsity_levels)
    pbar = tqdm(total=total, desc="Analyzing")

    for sample in samples:
        context = sample['context']
        question = sample['question']
        answer = sample['answer']

        # Truncate context if too long
        context_words = context.split()
        if len(context_words) > 800:
            context = ' '.join(context_words[:800])

        for sparsity in sparsity_levels:
            for method in methods:
                atas = 0.0
                try:
                    atas = analyzer.compute_option_atas(
                        context, question, answer,
                        method=method,
                        sparsity=sparsity
                    )
                    results[method][sparsity].append(atas)

                except Exception as e:
                    print(f"\nError: {e}")
                    results[method][sparsity].append(0.0)

                pbar.set_postfix({
                    'method': method,
                    'sparsity': f'{sparsity:.0%}',
                    'atas': f'{atas:.3f}'
                })
                pbar.update(1)

    pbar.close()

    # Summary
    summary = {}
    for method in methods:
        summary[method] = {}
        for sparsity in sparsity_levels:
            avg_atas = np.mean(results[method][sparsity])
            std_atas = np.std(results[method][sparsity])
            summary[method][sparsity] = {
                'mean': avg_atas,
                'std': std_atas
            }

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS: QuALITY ATTENTION PRESERVATION")
    print("=" * 80)

    for sparsity in sparsity_levels:
        print(f"\nSparsity: {sparsity:.0%}")
        print(f"{'Method':<10} | {'Mean ATAS':<12} | {'Std':<10}")
        print("-" * 45)

        for method in methods:
            mean = summary[method][sparsity]['mean']
            std = summary[method][sparsity]['std']
            print(f"{method.upper():<10} | {mean:>11.4f} | {std:>9.4f}")

    # Save
    with open('quality_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n✓ Saved results to quality_results.json")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  ATAS = Attention preserved to correct answer option")
    print("  Higher ATAS = Better preservation of answer information")
    print("  Expected: CAB > H2O at high sparsity (95-99%)")
    print("=" * 80)

    return summary


if __name__ == '__main__':
    results = run_quality_test(
        num_samples=20,
        methods=['full', 'h2o', 'cab'],
        sparsity_levels=[0.90, 0.95, 0.99],
        device='cuda'
    )
