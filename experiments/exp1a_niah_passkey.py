"""
Experiment 1A: Needle-in-a-Haystack (Passkey Retrieval)

The killer experiment for ICML: Proves that CAB-Attention preserves low-magnitude,
high-importance "bridges" that H2O drops.

Setup:
- Insert a passkey at varying depths in long documents
- Query: "What is the secret key?"
- Compare CAB vs H2O at different sparsity levels
- Generate accuracy heatmap
"""

import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import random
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from cab_attention.coarse_predictor import CoarseCurvaturePredictor
from cab_attention.kernels.frc_kernel import generate_block_mask_from_frc


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_passkey() -> str:
    """Generate a random 5-digit passkey."""
    return f"{random.randint(10000, 99999)}"


def generate_filler_text(num_tokens: int) -> str:
    """
    Generate filler text (haystack).
    Uses repeated common sentences to simulate irrelevant context.
    """
    sentences = [
        "The sky is blue and the grass is green.",
        "Water flows down the river to the sea.",
        "Birds fly south for the winter months.",
        "The sun rises in the east every morning.",
        "Mountains tower over the valleys below.",
        "Forests are home to many wild animals.",
        "Rain falls gently on the rooftop.",
        "Stars twinkle brightly in the night sky.",
        "Flowers bloom beautifully in the spring.",
        "The ocean waves crash against the shore.",
    ]

    # Approximate tokens (1 sentence ≈ 10 tokens)
    num_sentences = num_tokens // 10
    filler = " ".join([random.choice(sentences) for _ in range(num_sentences)])
    return filler


def create_niah_sample(
    context_length: int,
    needle_depth: float,  # 0.0 to 1.0
    passkey: str = None
) -> Dict[str, str]:
    """
    Create a single NIAH sample.

    Args:
        context_length: Total context in tokens (approximate)
        needle_depth: Where to insert passkey (0.0=start, 1.0=end)
        passkey: The secret key (if None, generate random)

    Returns:
        {
            'context': Full document with embedded passkey
            'needle': The needle sentence
            'passkey': The secret key
            'query': The question
            'answer': Expected answer
            'needle_position': Actual position in tokens
        }
    """
    if passkey is None:
        passkey = generate_passkey()

    # Create needle sentence
    needle = f"The secret key is: {passkey}"

    # Calculate position
    needle_tokens = len(needle.split())
    filler_tokens = context_length - needle_tokens
    needle_position = int(filler_tokens * needle_depth)

    # Generate filler before and after
    filler_before = generate_filler_text(needle_position)
    filler_after = generate_filler_text(filler_tokens - needle_position)

    # Combine
    context = f"{filler_before} {needle} {filler_after}"

    # Query and answer
    query = "What is the secret key?"
    answer = passkey

    return {
        'context': context,
        'needle': needle,
        'passkey': passkey,
        'query': query,
        'answer': answer,
        'needle_position': needle_position,
        'context_length': len(context.split()),  # Actual token count
    }


# ============================================================================
# Simplified Retrieval Model
# ============================================================================

class SimpleRetrievalModel:
    """
    Simplified model for NIAH testing.

    Instead of a full LLM, we use a simple retrieval mechanism:
    - Encode context and query as random embeddings
    - Compute attention scores
    - Apply sparsity
    - Check if the needle is in the attended tokens
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        device: str = 'cuda'
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device

    def tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization."""
        return text.lower().split()

    def embed(self, tokens: List[str]) -> torch.Tensor:
        """
        Create embeddings for tokens.
        For simplicity, we use random but consistent embeddings.
        """
        # Create a simple hash-based embedding
        embeddings = []
        for token in tokens:
            # Use hash for consistency
            seed = hash(token) % (2**32)
            torch.manual_seed(seed)
            emb = torch.randn(self.embed_dim)
            embeddings.append(emb)

        return torch.stack(embeddings).to(self.device)

    def retrieve(
        self,
        context: str,
        query: str,
        needle_tokens: List[str],
        attention_mask: torch.Tensor = None
    ) -> bool:
        """
        Check if retrieval is successful.

        Args:
            context: Full document
            query: Query string
            needle_tokens: Tokens in the needle
            attention_mask: [N, N] boolean mask (True = keep)

        Returns:
            success: True if needle tokens are in attended set
        """
        # Tokenize
        context_tokens = self.tokenize(context)

        # Find needle position
        needle_positions = []
        for i in range(len(context_tokens) - len(needle_tokens) + 1):
            if context_tokens[i:i+len(needle_tokens)] == needle_tokens:
                needle_positions = list(range(i, i + len(needle_tokens)))
                break

        if not needle_positions:
            return False  # Needle not in context (shouldn't happen)

        # If no mask, assume all tokens attended
        if attention_mask is None:
            return True

        # Check if ANY needle token is attended
        # In the last row (query attends to context)
        N = len(context_tokens)
        if attention_mask.shape[0] < N:
            # Pad mask if needed
            return False

        query_attention = attention_mask[-1, :]  # Last token attends to context

        # Check if any needle position is attended
        for pos in needle_positions:
            if pos < len(query_attention) and query_attention[pos]:
                return True

        return False


# ============================================================================
# Attention Methods
# ============================================================================

def compute_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute attention affinity matrix.

    Args:
        q: [1, H, N, D]
        k: [1, H, N, D]

    Returns:
        scores: [1, H, N, N]
    """
    scale = temperature / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    return torch.softmax(scores, dim=-1)


def apply_h2o_mask(
    attention_scores: torch.Tensor,
    sparsity: float,
    block_size: int = 64
) -> torch.Tensor:
    """
    H2O: Keep top-k blocks by magnitude.

    Args:
        attention_scores: [1, H, N, N]
        sparsity: Fraction to prune
        block_size: Block size for grouping

    Returns:
        mask: [N, N] boolean mask
    """
    B, H, N, _ = attention_scores.shape

    # Average across heads
    scores_avg = attention_scores.mean(dim=1)[0]  # [N, N]

    # Block-wise max (reduce to block level)
    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=scores_avg.device)

    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = scores_avg[i_start:i_end, j_start:j_end].max()

    # Keep top-k blocks by magnitude
    k_keep = max(1, int(M * M * (1 - sparsity)))
    threshold = torch.topk(block_scores.flatten(), k_keep).values[-1]
    block_mask = block_scores >= threshold

    # Expand to token level
    token_mask = torch.zeros(N, N, dtype=torch.bool, device=scores_avg.device)
    for i in range(M):
        for j in range(M):
            if block_mask[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask


def apply_cab_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    sparsity: float,
    block_size: int = 64,
    predictor: CoarseCurvaturePredictor = None
) -> torch.Tensor:
    """
    CAB: Keep blocks with lowest FRC (bridges).

    Args:
        q: [1, H, N, D]
        k: [1, H, N, D]
        sparsity: Fraction to prune
        block_size: Block size
        predictor: CoarseCurvaturePredictor instance

    Returns:
        mask: [N, N] boolean mask
    """
    if predictor is None:
        predictor = CoarseCurvaturePredictor(
            block_size=block_size,
            sparsity=sparsity,
        ).to(q.device)

    # Get block mask from predictor
    block_mask = predictor(q, k, return_diagnostics=False)  # [1, H, M, M]

    # Average across heads and remove batch dim
    block_mask_avg = block_mask.float().mean(dim=1)[0] > 0.5  # [M, M]

    # Expand to token level
    M = block_mask_avg.shape[0]
    N = q.shape[2]
    token_mask = torch.zeros(N, N, dtype=torch.bool, device=q.device)

    for i in range(M):
        for j in range(M):
            if block_mask_avg[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask


# ============================================================================
# Evaluation Harness
# ============================================================================

def evaluate_single_sample(
    sample: Dict,
    method: str,  # 'full', 'h2o', 'cab'
    sparsity: float,
    model: SimpleRetrievalModel,
    predictor: CoarseCurvaturePredictor = None,
    block_size: int = 64,
    device: str = 'cuda'
) -> bool:
    """
    Evaluate a single NIAH sample.

    Returns:
        success: True if retrieval succeeded
    """
    # Tokenize
    context_tokens = model.tokenize(sample['context'])
    needle_tokens = model.tokenize(sample['needle'])

    N = len(context_tokens)

    # Embed (simplified: random but consistent)
    q = model.embed(context_tokens).unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
    k = q.clone()  # Same for simplicity

    # Expand to multi-head
    q = q.expand(1, model.num_heads, N, model.embed_dim)
    k = k.expand(1, model.num_heads, N, model.embed_dim)

    # Compute attention scores
    attention_scores = compute_attention_scores(q, k)

    # Apply masking based on method
    if method == 'full':
        mask = torch.ones(N, N, dtype=torch.bool, device=device)
    elif method == 'h2o':
        mask = apply_h2o_mask(attention_scores, sparsity, block_size)
    elif method == 'cab':
        mask = apply_cab_mask(q, k, sparsity, block_size, predictor)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Check retrieval
    success = model.retrieve(sample['context'], sample['query'], needle_tokens, mask)

    return success


def run_niah_experiment(
    context_lengths: List[int],
    needle_depths: List[float],
    sparsity_levels: List[float],
    methods: List[str],
    num_samples: int = 5,
    block_size: int = 64,
    device: str = 'cuda'
) -> Dict:
    """
    Run full NIAH experiment.

    Returns:
        results: Nested dict with accuracies
    """
    model = SimpleRetrievalModel(device=device)
    predictor = CoarseCurvaturePredictor(block_size=block_size, sparsity=0.95).to(device)

    results = {method: {} for method in methods}

    print("=" * 80)
    print("EXPERIMENT 1A: NEEDLE-IN-A-HAYSTACK (PASSKEY RETRIEVAL)")
    print("=" * 80)
    print(f"Context lengths: {context_lengths}")
    print(f"Needle depths: {needle_depths}")
    print(f"Sparsity levels: {sparsity_levels}")
    print(f"Methods: {methods}")
    print(f"Samples per config: {num_samples}")
    print("=" * 80)

    total_configs = len(context_lengths) * len(needle_depths) * len(sparsity_levels) * len(methods)
    pbar = tqdm(total=total_configs, desc="Running NIAH")

    for ctx_len in context_lengths:
        for depth in needle_depths:
            for sparsity in sparsity_levels:
                # Generate samples
                samples = [create_niah_sample(ctx_len, depth) for _ in range(num_samples)]

                for method in methods:
                    # Evaluate
                    successes = []
                    for sample in samples:
                        try:
                            success = evaluate_single_sample(
                                sample, method, sparsity, model, predictor, block_size, device
                            )
                            successes.append(success)
                        except Exception as e:
                            print(f"\nError with {method}, N={ctx_len}, depth={depth}, sparsity={sparsity}: {e}")
                            successes.append(False)

                    # Compute accuracy
                    accuracy = sum(successes) / len(successes)

                    # Store results
                    key = (ctx_len, depth, sparsity)
                    results[method][key] = accuracy

                    pbar.set_postfix({
                        'method': method,
                        'N': ctx_len,
                        'depth': depth,
                        'sparsity': sparsity,
                        'acc': f'{accuracy:.2f}'
                    })
                    pbar.update(1)

    pbar.close()
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_niah_heatmap(
    results: Dict,
    context_lengths: List[int],
    needle_depths: List[float],
    sparsity: float,
    save_path: str = 'niah_heatmap.png'
):
    """
    Generate the killer heatmap: accuracy vs (context length, needle depth).
    """
    methods = list(results.keys())
    num_methods = len(methods)

    fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 5))
    if num_methods == 1:
        axes = [axes]

    for idx, method in enumerate(methods):
        # Create matrix
        matrix = np.zeros((len(needle_depths), len(context_lengths)))

        for i, depth in enumerate(needle_depths):
            for j, ctx_len in enumerate(context_lengths):
                key = (ctx_len, depth, sparsity)
                matrix[i, j] = results[method].get(key, 0.0)

        # Plot
        sns.heatmap(
            matrix,
            ax=axes[idx],
            xticklabels=[f'{n//1000}k' for n in context_lengths],
            yticklabels=[f'{int(d*100)}%' for d in needle_depths],
            cmap='RdYlGn',
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Accuracy'}
        )
        axes[idx].set_title(f'{method.upper()} (Sparsity={sparsity:.0%})', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Context Length', fontsize=12)
        axes[idx].set_ylabel('Needle Depth', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved heatmap: {save_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run Experiment 1A."""

    # Configuration
    context_lengths = [1024, 2048, 4096, 8192]  # Start small, scale up
    needle_depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    sparsity_levels = [0.90, 0.95, 0.99]
    methods = ['full', 'h2o', 'cab']
    num_samples = 3  # Small for speed

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Run experiment
    results = run_niah_experiment(
        context_lengths=context_lengths,
        needle_depths=needle_depths,
        sparsity_levels=sparsity_levels,
        methods=methods,
        num_samples=num_samples,
        device=device
    )

    # Save results
    results_serializable = {
        method: {str(k): v for k, v in method_results.items()}
        for method, method_results in results.items()
    }

    with open('niah_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print("\nSaved results: niah_results.json")

    # Generate heatmaps for each sparsity level
    for sparsity in sparsity_levels:
        plot_niah_heatmap(
            results,
            context_lengths,
            needle_depths,
            sparsity,
            save_path=f'niah_heatmap_sparsity_{int(sparsity*100)}.png'
        )

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
    print("Experiment 1A completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
