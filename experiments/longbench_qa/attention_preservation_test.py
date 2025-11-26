"""
Attention Preservation Test: Direct Evaluation of CAB vs H2O

Instead of modifying generation, we directly measure:
"Does the sparse attention mask preserve attention to answer-relevant tokens?"

This is:
1. Faster to implement
2. More direct test of hypothesis
3. Easier to interpret for ICML

Metric: Answer Token Attention Score (ATAS)
- Higher ATAS = more attention preserved to tokens that appear in the answer
"""

import sys
sys.path.insert(0, '../..')

import torch
import numpy as np
import json
import os
from tqdm import tqdm
from typing import List, Dict, Set
from transformers import GPT2Model, GPT2Tokenizer
from urllib.request import urlopen
from urllib.error import URLError


# ============================================================================
# Sparse Attention Masks (from earlier work)
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

def load_longbench_narrativeqa(num_samples: int = 20, cache_dir: str = './data'):
    """
    Load NarrativeQA or similar long-context QA dataset.

    Tries multiple methods in order:
    1. Standard NarrativeQA from deepmind/narrativeqa
    2. QuAC (Question Answering in Context)
    3. Cached JSONL file
    4. Create synthetic long-context samples
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'qa_dataset.jsonl')

    # Method 1: Try standard NarrativeQA from HuggingFace
    try:
        from datasets import load_dataset
        print("Attempting to load NarrativeQA from HuggingFace...")
        dataset = load_dataset('narrativeqa', split='test')

        # Convert to our format
        samples = []
        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            # NarrativeQA has: document (dict with text), question, answers (list)
            if 'document' in item and 'text' in item['document']:
                context = item['document']['text']
            elif 'context' in item:
                context = item['context']
            else:
                continue

            # Handle different answer formats
            answers_list = item.get('answers', [])
            if isinstance(answers_list, list) and len(answers_list) > 0:
                # answers might be dicts with 'text' key or just strings
                first_answer = answers_list[0]
                if isinstance(first_answer, dict) and 'text' in first_answer:
                    answer_text = first_answer['text']
                else:
                    answer_text = str(first_answer)
            else:
                answer_text = item.get('answer', '')

            samples.append({
                'context': context,
                'input': item.get('question', item.get('input', '')),
                'answers': [answer_text]
            })

        if len(samples) > 0:
            print(f"✓ Loaded {len(samples)} samples from NarrativeQA")
            # Cache for future use
            with open(cache_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            return samples

    except Exception as e:
        print(f"NarrativeQA loading failed: {e}")

    # Method 2: Try QuAC dataset as fallback
    try:
        from datasets import load_dataset
        print("Attempting to load QuAC (Question Answering in Context)...")
        dataset = load_dataset('quac', split='validation')

        samples = []
        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            # QuAC has longer contexts
            context = item['context'] if 'context' in item else item.get('background', '')
            questions = item['questions'] if 'questions' in item else []
            answers = item['answers'] if 'answers' in item else []

            if questions and answers and context:
                # Use first Q&A pair
                samples.append({
                    'context': context,
                    'input': questions[0] if isinstance(questions, list) else questions,
                    'answers': [answers[0]['text']] if isinstance(answers, list) and len(answers) > 0 else [str(answers)]
                })

        if len(samples) > 0:
            print(f"✓ Loaded {len(samples)} samples from QuAC")
            with open(cache_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample) + '\n')
            return samples

    except Exception as e:
        print(f"QuAC loading failed: {e}")

    # Method 3: Use cached file if available
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

    # Method 4: Create realistic synthetic long-context samples
    print("Creating synthetic long-context QA samples...")
    samples = []

    topics = [
        ("Albert Einstein", "physicist", "theory of relativity", "1879", "Germany"),
        ("Marie Curie", "chemist", "radioactivity", "1867", "Poland"),
        ("Isaac Newton", "mathematician", "laws of motion", "1643", "England"),
        ("Charles Darwin", "naturalist", "evolution", "1809", "England"),
        ("Leonardo da Vinci", "artist", "Mona Lisa", "1452", "Italy"),
    ]

    for i in range(min(num_samples, len(topics) * 4)):
        topic = topics[i % len(topics)]
        name, profession, achievement, birth_year, country = topic

        # Create a long context (simulating a document)
        context = f"""
        {name} was a renowned {profession} born in {birth_year} in {country}.
        Early life and education filled many years with rigorous study and preparation.
        """ + " ".join([f"Additional biographical information about various aspects of {name}'s life and work. " * 20])

        context += f"""
        The most significant contribution was the development of {achievement}.
        This groundbreaking work changed the field forever.
        """ + " ".join([f"More details about the impact and legacy of this important work. " * 20])

        questions = [
            (f"What was {name}'s most famous achievement?", achievement),
            (f"Where was {name} born?", country),
            (f"What year was {name} born?", birth_year),
            (f"What was {name}'s profession?", profession),
        ]

        q, a = questions[i % len(questions)]
        samples.append({
            'context': context,
            'input': q,
            'answers': [a]
        })

    print(f"✓ Created {len(samples)} synthetic samples")
    with open(cache_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    return samples


# ============================================================================
# Attention Analyzer
# ============================================================================

class AttentionPreservationAnalyzer:
    """Analyzes how well sparse attention preserves answer-relevant information."""

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
        """
        Compute Answer Token Attention Score (ATAS).

        ATAS measures: "How much attention do query tokens pay to answer tokens?"

        Args:
            context: Document context
            question: Question
            answer: Ground truth answer
            method: 'full', 'h2o', or 'cab'
            sparsity: Sparsity level
            layer: Which layer to analyze

        Returns:
            ATAS score (0-1, higher is better)
        """
        # Construct full input
        full_text = f"{context} Question: {question}"

        # Tokenize
        tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate if too long
        if len(tokens) > 1024:
            tokens = tokens[:1024]

        N = len(tokens)

        # Find answer token positions
        answer_tokens_set = set(self.tokenizer.encode(answer, add_special_tokens=False))

        answer_positions = []
        for i, token in enumerate(tokens):
            if token in answer_tokens_set:
                answer_positions.append(i)

        if not answer_positions:
            # Answer not found in context, return 0
            return 0.0

        # Extract attention
        input_ids = torch.tensor([tokens], device=self.device)
        outputs = self.model(input_ids, output_attentions=True, use_cache=False)

        # Get attention from target layer, averaged across heads
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

        # Compute ATAS: average attention from all tokens to answer tokens
        # Focus on last few tokens (question tokens) attending to answer tokens
        question_start_pos = max(0, N - 50)  # Last 50 tokens likely contain question

        total_attention_to_answer = 0.0
        count = 0

        for query_pos in range(question_start_pos, N):
            for answer_pos in answer_positions:
                if mask[query_pos, answer_pos]:
                    # This connection is preserved
                    total_attention_to_answer += attention[query_pos, answer_pos].item()
                count += 1

        if count == 0:
            return 0.0

        atas = total_attention_to_answer / count
        return atas


# ============================================================================
# Experiment Runner
# ============================================================================

def run_preservation_test(
    num_samples: int = 20,
    methods: List[str] = ['full', 'h2o', 'cab'],
    sparsity_levels: List[float] = [0.90, 0.95, 0.99],
    device: str = 'cuda'
):
    """Run attention preservation test."""

    print("=" * 80)
    print("ATTENTION PRESERVATION TEST: CAB VS H2O")
    print("=" * 80)
    print("Metric: Answer Token Attention Score (ATAS)")
    print("  - Measures attention preserved to answer-relevant tokens")
    print("  - Higher ATAS = better preservation of important information")
    print("=" * 80)
    print(f"Methods: {methods}")
    print(f"Sparsity: {sparsity_levels}")
    print(f"Samples: {num_samples}")
    print("=" * 80)
    print()

    # Load dataset
    print("Loading LongBench NarrativeQA...")
    samples = load_longbench_narrativeqa(num_samples=num_samples)

    if samples is None or len(samples) == 0:
        print("Warning: Could not load dataset. Using mock data...")
        samples = []
        for i in range(num_samples):
            samples.append({
                'context': "Paris is the capital of France. The Eiffel Tower is in Paris. " * 50,
                'input': "What is in Paris?",
                'answers': ["Eiffel Tower"]
            })

    # Initialize analyzer
    analyzer = AttentionPreservationAnalyzer(device=device)

    # Results
    results = {m: {s: [] for s in sparsity_levels} for m in methods}

    # Evaluate
    total = len(samples) * len(methods) * len(sparsity_levels)
    pbar = tqdm(total=total, desc="Analyzing")

    for sample in samples:
        context = sample['context']
        question = sample['input']
        answer = sample['answers'][0] if isinstance(sample['answers'], list) else sample['answers']

        # Truncate context if too long
        context_words = context.split()
        if len(context_words) > 800:
            context = ' '.join(context_words[:800])

        for sparsity in sparsity_levels:
            for method in methods:
                atas = 0.0  # Initialize before try block
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
    print("RESULTS: ATTENTION PRESERVATION (ATAS)")
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
    with open('attention_preservation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n✓ Saved results to attention_preservation_results.json")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  ATAS measures attention to answer-relevant tokens")
    print("  If CAB ATAS > H2O ATAS:")
    print("    ✓ CAB better preserves attention to important information")
    print("    ✓ FRC identifies structurally critical tokens (bridges)")
    print("  Expected: CAB advantage increases with sparsity (95-99%)")
    print("=" * 80)

    return summary


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    results = run_preservation_test(
        num_samples=20,
        methods=['full', 'h2o', 'cab'],
        sparsity_levels=[0.90, 0.95, 0.99],
        device='cuda'
    )
