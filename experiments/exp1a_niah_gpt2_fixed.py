"""
Experiment 1A: GPT-2 NIAH - FIXED VERSION

Key fix: More robust needle detection using passkey-only search
"""

import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import random
import json
from tqdm import tqdm
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import GPT2Model, GPT2Tokenizer


# ============================================================================
# Improved Needle Dataset
# ============================================================================

class FixedNeedleDataset:
    """Fixed needle detection using passkey tokens only."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.filler_sentences = [
            "The sky is blue and the grass is green.",
            "Water flows down the river to the sea.",
            "Birds fly south for the winter months.",
            "The sun rises in the east every morning.",
            "Mountains tower over the valleys below.",
        ]

    def generate_passkey(self) -> str:
        return f"{random.randint(10000, 99999)}"

    def generate_filler_text(self, target_tokens: int) -> str:
        num_sentences = (target_tokens // 12) + 1
        sentences = [random.choice(self.filler_sentences) for _ in range(num_sentences)]
        return " ".join(sentences)

    def create_sample(
        self,
        context_length: int,
        needle_depth: float
    ) -> Dict:
        """Create NIAH sample with ROBUST needle detection."""
        passkey = self.generate_passkey()

        # FIX: Use a simpler needle format that's easier to detect
        # Just the number surrounded by distinctive text
        needle_text = f" PASSKEY {passkey} "  # Spaces help tokenization

        # Generate filler (make it shorter to leave room for needle)
        filler_tokens_target = context_length - 20  # Leave margin
        needle_position = int(filler_tokens_target * needle_depth)

        filler_before = self.generate_filler_text(needle_position)
        filler_after = self.generate_filler_text(filler_tokens_target - needle_position)

        # Construct context
        context = f"{filler_before}{needle_text}{filler_after}"

        # Tokenize
        context_token_ids = self.tokenizer.encode(context, add_special_tokens=False)

        # FIX: Search for passkey tokens ONLY (most robust)
        passkey_tokens = self.tokenizer.encode(passkey, add_special_tokens=False)

        needle_positions = []
        for i in range(len(context_token_ids) - len(passkey_tokens) + 1):
            # Check if all passkey tokens match
            match = True
            for j, pk_token in enumerate(passkey_tokens):
                if i + j >= len(context_token_ids) or context_token_ids[i + j] != pk_token:
                    match = False
                    break

            if match:
                needle_positions = list(range(i, i + len(passkey_tokens)))
                break

        return {
            'context_token_ids': context_token_ids,
            'needle_positions': needle_positions,
            'passkey': passkey,
            'actual_length': len(context_token_ids),
        }


# ============================================================================
# Attention Extraction
# ============================================================================

class SimpleAttentionExtractor:
    """Simplified attention extractor."""

    def __init__(self, device='cuda'):
        self.device = device
        print("Loading GPT-2...")
        self.model = GPT2Model.from_pretrained(
            "gpt2",
            attn_implementation='eager'
        ).to(device)
        self.model.eval()
        print(f"✓ Loaded: 12 layers, 12 heads")

    @torch.no_grad()
    def extract_attention(self, token_ids: List[int]) -> torch.Tensor:
        """Extract attention from layer 6, averaged across heads."""
        input_ids = torch.tensor([token_ids], device=self.device)
        outputs = self.model(input_ids, output_attentions=True, use_cache=False)
        attention = outputs.attentions[6]  # Layer 6
        return attention.mean(dim=1)[0]  # Average across heads → [N, N]


# ============================================================================
# Sparse Masks (Same as before)
# ============================================================================

def apply_h2o_mask(attention: torch.Tensor, sparsity: float, block_size: int = 64) -> torch.Tensor:
    """H2O: Top-k by magnitude."""
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


def apply_cab_mask(attention: torch.Tensor, sparsity: float, block_size: int = 64) -> torch.Tensor:
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
    frc_scores = block_scores - 0.5 * redundancy

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
# Evaluation
# ============================================================================

def evaluate_sample(
    sample: Dict,
    method: str,
    sparsity: float,
    extractor: SimpleAttentionExtractor
) -> bool:
    """Evaluate single sample."""
    token_ids = sample['context_token_ids']
    needle_positions = sample['needle_positions']

    if not needle_positions:
        # No needle found - this is a bug, but return False
        return False

    # Extract attention
    dense_attention = extractor.extract_attention(token_ids)
    N = dense_attention.shape[0]

    # Apply sparsity
    if method == 'full':
        mask = torch.ones(N, N, dtype=torch.bool, device=dense_attention.device)
    elif method == 'h2o':
        mask = apply_h2o_mask(dense_attention, sparsity)
    elif method == 'cab':
        mask = apply_cab_mask(dense_attention, sparsity)
    else:
        raise ValueError(f"Unknown method: {method}")

    # FIX: Check if ANY position attends to ANY needle position
    # (In GPT-2, all tokens can attend to all previous tokens)
    for query_pos in range(N):
        for needle_pos in needle_positions:
            if needle_pos < N and mask[query_pos, needle_pos]:
                return True  # Found at least one attention to needle

    return False


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment():
    """Run fixed experiment."""
    print("=" * 80)
    print("EXPERIMENT 1A: GPT-2 NIAH (FIXED VERSION)")
    print("=" * 80)

    # Config
    context_lengths = [512, 1024]
    needle_depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    sparsity_levels = [0.90, 0.95, 0.99]
    methods = ['full', 'h2o', 'cab']
    num_samples = 5
    device = 'cuda'

    print(f"Context lengths: {context_lengths}")
    print(f"Needle depths: {needle_depths}")
    print(f"Sparsity levels: {sparsity_levels}")
    print("=" * 80)
    print()

    # Initialize
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataset = FixedNeedleDataset(tokenizer)
    extractor = SimpleAttentionExtractor(device)

    results = {method: {} for method in methods}

    total = len(context_lengths) * len(needle_depths) * len(sparsity_levels) * len(methods)
    pbar = tqdm(total=total, desc="Running")

    # First, let's validate needle detection works
    print("\nValidating needle detection...")
    test_sample = dataset.create_sample(512, 0.5)
    print(f"  Context length: {test_sample['actual_length']}")
    print(f"  Needle positions: {test_sample['needle_positions']}")
    print(f"  Needle found: {len(test_sample['needle_positions']) > 0}")
    print()

    for ctx_len in context_lengths:
        for depth in needle_depths:
            # Generate samples
            samples = [dataset.create_sample(ctx_len, depth) for _ in range(num_samples)]

            # Debug: check if needles were found
            found_count = sum(1 for s in samples if s['needle_positions'])
            if found_count == 0:
                print(f"\nWARNING: No needles found for N={ctx_len}, depth={depth}")

            for sparsity in sparsity_levels:
                for method in methods:
                    successes = []
                    for sample in samples:
                        try:
                            success = evaluate_sample(sample, method, sparsity, extractor)
                            successes.append(success)
                        except Exception as e:
                            print(f"\nError: {e}")
                            successes.append(False)

                    accuracy = sum(successes) / len(successes)
                    results[method][(ctx_len, depth, sparsity)] = accuracy

                    pbar.set_postfix({
                        'method': method,
                        'N': ctx_len,
                        'depth': f'{depth:.2f}',
                        'sparsity': f'{sparsity:.0%}',
                        'acc': f'{accuracy:.2f}'
                    })
                    pbar.update(1)

    pbar.close()

    # Save results
    results_serializable = {
        method: {str(k): v for k, v in r.items()}
        for method, r in results.items()
    }

    with open('niah_results_gpt2_fixed.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for sparsity in sparsity_levels:
        print(f"\nSparsity: {sparsity:.0%}")
        print(f"{'Method':<10} | {'Avg Accuracy':<15} | {'Min Accuracy':<15}")
        print("-" * 50)

        for method in methods:
            accs = [v for k, v in results[method].items() if k[2] == sparsity]
            if accs:
                avg_acc = np.mean(accs)
                min_acc = np.min(accs)
                print(f"{method.upper():<10} | {avg_acc:>14.2%} | {min_acc:>14.2%}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    run_experiment()
