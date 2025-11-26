"""
Qasper Attention Preservation Test

Qasper: Question Answering on Scientific Papers
- Questions about NLP research papers
- Mix of extractive and abstractive answers
- Long technical contexts
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

def load_qasper_dataset(num_samples: int = 20, cache_dir: str = './data'):
    """Load Qasper dataset from HuggingFace."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'qasper.jsonl')

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset
        print("Loading Qasper from HuggingFace...")
        dataset = load_dataset('qasper', split='validation')

        samples = []
        for i in range(min(num_samples * 5, len(dataset))):  # Sample more since structure is nested
            item = dataset[i]

            # Qasper has complex structure: full_text (dict), qas (dict with questions)
            # Concatenate abstract and full_text
            abstract = item.get('abstract', '')
            full_text_parts = item.get('full_text', {})

            # Build context from paragraphs
            if isinstance(full_text_parts, dict) and 'paragraphs' in full_text_parts:
                paragraphs = full_text_parts['paragraphs']
                if isinstance(paragraphs, list):
                    context = abstract + " " + " ".join([p for p in paragraphs if isinstance(p, str)][:10])  # First 10 paragraphs
                else:
                    context = abstract
            else:
                context = abstract

            # Extract Q&A pairs
            qas = item.get('qas', {})
            if isinstance(qas, dict) and 'questions' in qas:
                questions = qas.get('questions', [])
                answers = qas.get('answers', [])

                for q, a in zip(questions[:2], answers[:2]):  # Max 2 Q&A pairs per paper
                    if len(samples) >= num_samples:
                        break

                    # answers is usually a list of dicts with 'answer' field
                    if isinstance(a, list) and len(a) > 0:
                        if isinstance(a[0], dict) and 'answer' in a[0]:
                            answer_text = a[0]['answer']
                        else:
                            answer_text = str(a[0])
                    else:
                        answer_text = str(a)

                    samples.append({
                        'context': context,
                        'question': q if isinstance(q, str) else str(q),
                        'answer': answer_text
                    })

            if len(samples) >= num_samples:
                break

        if len(samples) > 0:
            print(f"✓ Loaded {len(samples)} samples from Qasper")
            with open(cache_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            return samples

    except Exception as e:
        print(f"Qasper loading failed: {e}")
        import traceback
        traceback.print_exc()

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

    print("Could not load Qasper dataset")
    return None


# ============================================================================
# Attention Analyzer
# ============================================================================

class QasperAttentionAnalyzer:
    """Analyzes attention preservation for Qasper scientific paper QA."""

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
        method: str = 'full',
        sparsity: float = 0.95,
        layer: int = 6
    ) -> float:
        """Compute ATAS for answer tokens."""
        # Construct input
        full_text = f"{context} Question: {question}"

        # Tokenize
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate if needed
        if len(tokens) > 1024:
            tokens = tokens[:1024]

        N = len(tokens)

        # Find answer tokens
        answer_tokens_set = set(self.tokenizer.encode(answer, add_special_tokens=False))

        answer_positions = []
        for i, token in enumerate(tokens):
            if token in answer_tokens_set:
                answer_positions.append(i)

        if not answer_positions:
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


# ============================================================================
# Experiment Runner
# ============================================================================

def run_qasper_test(
    num_samples: int = 20,
    methods: List[str] = ['full', 'h2o', 'cab'],
    sparsity_levels: List[float] = [0.90, 0.95, 0.99],
    device: str = 'cuda'
):
    """Run Qasper attention preservation test."""

    print("=" * 80)
    print("QASPER ATTENTION PRESERVATION TEST")
    print("=" * 80)
    print("Dataset: Qasper (Scientific Paper QA)")
    print("Metric: Answer Token Attention Score (ATAS)")
    print("=" * 80)
    print(f"Methods: {methods}")
    print(f"Sparsity: {sparsity_levels}")
    print(f"Samples: {num_samples}")
    print("=" * 80)
    print()

    # Load dataset
    samples = load_qasper_dataset(num_samples=num_samples)

    if samples is None or len(samples) == 0:
        print("ERROR: Could not load Qasper dataset")
        return None

    # Initialize analyzer
    analyzer = QasperAttentionAnalyzer(device=device)

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
                    atas = analyzer.compute_atas(
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
    print("RESULTS: QASPER ATTENTION PRESERVATION")
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
    with open('qasper_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n✓ Saved results to qasper_results.json")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  ATAS = Attention preserved to answer tokens")
    print("  Higher ATAS = Better preservation of answer information")
    print("  Expected: CAB > H2O at high sparsity (95-99%)")
    print("=" * 80)

    return summary


if __name__ == '__main__':
    results = run_qasper_test(
        num_samples=20,
        methods=['full', 'h2o', 'cab'],
        sparsity_levels=[0.90, 0.95, 0.99],
        device='cuda'
    )
