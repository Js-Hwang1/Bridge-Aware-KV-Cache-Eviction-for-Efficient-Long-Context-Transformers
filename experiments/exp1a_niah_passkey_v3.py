"""
Experiment 1A: Needle-in-a-Haystack (Passkey Retrieval) - Version 3

PATH A FIX: Add query-needle affinity
- Query token explicitly attends to needle (not random)
- This makes dense attention highlight the needle
- Then we can test if CAB preserves this signal better than H2O

KEY CHANGE: Add a query token with high similarity to needle_prototype
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

from cab_attention.kernels.coarsening import coarsen_qk_max_l2_pytorch
from cab_attention.kernels.frc_kernel import compute_block_frc, generate_block_mask_from_frc


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_passkey() -> str:
    """Generate a random 5-digit passkey."""
    return f"{random.randint(10000, 99999)}"


def generate_filler_text(num_tokens: int) -> str:
    """Generate filler text (haystack)."""
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
    num_sentences = num_tokens // 10
    filler = " ".join([random.choice(sentences) for _ in range(num_sentences)])
    return filler


def create_niah_sample(
    context_length: int,
    needle_depth: float,
    passkey: str = None
) -> Dict[str, str]:
    """Create a single NIAH sample."""
    if passkey is None:
        passkey = generate_passkey()

    needle = f"The secret key is: {passkey}"
    needle_tokens = len(needle.split())
    filler_tokens = context_length - needle_tokens
    needle_position = int(filler_tokens * needle_depth)

    filler_before = generate_filler_text(needle_position)
    filler_after = generate_filler_text(filler_tokens - needle_position)

    context = f"{filler_before} {needle} {filler_after}"

    return {
        'context': context,
        'needle': needle,
        'passkey': passkey,
        'query': "What is the secret key?",
        'answer': passkey,
        'needle_position': needle_position,
        'context_length': len(context.split()),
    }


# ============================================================================
# PATH A FIX: Query-Needle Affinity Model
# ============================================================================

class QueryAwareRetrievalModel:
    """
    PATH A FIX: Add explicit query-needle affinity.

    Key improvements:
    1. Filler tokens: clustered (redundant)
    2. Needle tokens: distinct (bridge-like)
    3. **Query token: high affinity to needle** (NEW!)

    This ensures dense attention naturally highlights the needle,
    giving FRC meaningful structure to analyze.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 1,
        device: str = 'cuda'
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device

        # Create orthogonal prototypes
        self.filler_prototype = torch.randn(embed_dim, device=device)
        self.needle_prototype = torch.randn(embed_dim, device=device)
        self.query_prototype = torch.randn(embed_dim, device=device)

        # Make needle orthogonal to filler
        self.needle_prototype = self.needle_prototype - (
            self.needle_prototype @ self.filler_prototype
        ) * self.filler_prototype

        # Make query SIMILAR to needle (high dot product)
        # Query = needle + small orthogonal component
        self.query_prototype = 0.9 * self.needle_prototype + 0.1 * self.filler_prototype

        # Normalize all
        self.needle_prototype = self.needle_prototype / self.needle_prototype.norm()
        self.filler_prototype = self.filler_prototype / self.filler_prototype.norm()
        self.query_prototype = self.query_prototype / self.query_prototype.norm()

        print(f"✓ Prototypes initialized")
        print(f"  - Needle·Filler similarity: {(self.needle_prototype @ self.filler_prototype).item():.3f}")
        print(f"  - Query·Needle similarity: {(self.query_prototype @ self.needle_prototype).item():.3f}")
        print(f"  - Query·Filler similarity: {(self.query_prototype @ self.filler_prototype).item():.3f}")

    def tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization."""
        return text.lower().split()

    def embed(self, tokens: List[str], needle_tokens: List[str]) -> torch.Tensor:
        """
        Create structured embeddings with query token.

        Returns:
            embeddings: [N+1, D] where last token is the query
        """
        embeddings = []
        needle_set = set(needle_tokens)

        # Context tokens
        for token in tokens:
            if token in needle_set:
                # Needle: distinct, based on needle_prototype
                noise = torch.randn(self.embed_dim, device=self.device) * 0.05
                emb = self.needle_prototype + noise
            else:
                # Filler: clustered, based on filler_prototype
                noise = torch.randn(self.embed_dim, device=self.device) * 0.2
                emb = self.filler_prototype + noise

            embeddings.append(emb / emb.norm())

        # Add query token at the end (attends to needle)
        query_noise = torch.randn(self.embed_dim, device=self.device) * 0.05
        query_emb = self.query_prototype + query_noise
        embeddings.append(query_emb / query_emb.norm())

        return torch.stack(embeddings)

    def compute_dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute full dense attention.

        Now the LAST row (query token) should have high attention to needle!

        Args:
            q: [B, H, N, D]
            k: [B, H, N, D]

        Returns:
            attention: [B, H, N, N] - softmax-normalized
        """
        scale = temperature / (q.shape[-1] ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attention = torch.softmax(scores, dim=-1)
        return attention

    def retrieve(
        self,
        context: str,
        query: str,
        needle_tokens: List[str],
        attention_mask: torch.Tensor = None
    ) -> bool:
        """
        Check if retrieval is successful.

        Success = query token (last position) attends to any needle token.
        """
        context_tokens = self.tokenize(context)

        # Find needle positions
        needle_positions = []
        for i in range(len(context_tokens) - len(needle_tokens) + 1):
            if context_tokens[i:i+len(needle_tokens)] == needle_tokens:
                needle_positions = list(range(i, i + len(needle_tokens)))
                break

        if not needle_positions:
            return False

        if attention_mask is None:
            return True

        N = len(context_tokens) + 1  # +1 for query token
        if attention_mask.shape[0] < N:
            return False

        # Query token is at position N-1 (last)
        query_attention = attention_mask[-1, :-1]  # Exclude self-attention to query

        # Success if query attends to any needle position
        for pos in needle_positions:
            if pos < len(query_attention) and query_attention[pos]:
                return True

        return False


# ============================================================================
# Sparse Attention Methods (Same as V2)
# ============================================================================

def apply_h2o_mask_v3(
    attention_scores: torch.Tensor,
    sparsity: float,
    block_size: int = 64
) -> torch.Tensor:
    """H2O: Keep top-k blocks by magnitude."""
    B, H, N, _ = attention_scores.shape

    scores_avg = attention_scores.mean(dim=1)[0]  # [N, N]

    # Blockify
    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=scores_avg.device)

    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = scores_avg[i_start:i_end, j_start:j_end].max()

    # Top-k by magnitude
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


def apply_cab_mask_v3(
    attention_scores: torch.Tensor,
    sparsity: float,
    block_size: int = 64
) -> torch.Tensor:
    """CAB: Keep blocks with LOWEST FRC (bridges)."""
    B, H, N, _ = attention_scores.shape

    scores_avg = attention_scores.mean(dim=1)[0]  # [N, N]

    # Blockify
    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=scores_avg.device)

    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = scores_avg[i_start:i_end, j_start:j_end].mean()

    # Compute FRC on real attention graph
    triangles = torch.matmul(block_scores, block_scores)  # 2-hop paths
    lambda_redundancy = 0.5
    frc_scores = block_scores - lambda_redundancy * triangles

    # Keep lowest FRC (bridges)
    k_keep = max(1, int(M * M * (1 - sparsity)))
    threshold = torch.topk(frc_scores.flatten(), k_keep, largest=False).values[-1]
    block_mask = frc_scores <= threshold

    # Expand to token level
    token_mask = torch.zeros(N, N, dtype=torch.bool, device=scores_avg.device)
    for i in range(M):
        for j in range(M):
            if block_mask[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask


# ============================================================================
# Evaluation Harness
# ============================================================================

def evaluate_single_sample_v3(
    sample: Dict,
    method: str,
    sparsity: float,
    model: QueryAwareRetrievalModel,
    block_size: int = 64,
    device: str = 'cuda'
) -> bool:
    """Evaluate with query-aware model."""
    context_tokens = model.tokenize(sample['context'])
    needle_tokens = model.tokenize(sample['needle'])

    # Create embeddings (includes query token at end)
    embeddings = model.embed(context_tokens, needle_tokens)
    N = embeddings.shape[0]  # N = context + 1 query

    # Q = K for self-attention
    q = embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
    k = q.clone()

    q = q.expand(1, model.num_heads, N, model.embed_dim)
    k = k.expand(1, model.num_heads, N, model.embed_dim)

    # Compute dense attention
    dense_attention = model.compute_dense_attention(q, k)

    # Apply sparsity
    if method == 'full':
        mask = torch.ones(N, N, dtype=torch.bool, device=device)
    elif method == 'h2o':
        mask = apply_h2o_mask_v3(dense_attention, sparsity, block_size)
    elif method == 'cab':
        mask = apply_cab_mask_v3(dense_attention, sparsity, block_size)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Check retrieval
    success = model.retrieve(sample['context'], sample['query'], needle_tokens, mask)

    return success


def run_niah_experiment_v3(
    context_lengths: List[int],
    needle_depths: List[float],
    sparsity_levels: List[float],
    methods: List[str],
    num_samples: int = 5,
    block_size: int = 64,
    device: str = 'cuda'
) -> Dict:
    """Run full NIAH experiment V3."""
    model = QueryAwareRetrievalModel(device=device)

    results = {method: {} for method in methods}

    print("=" * 80)
    print("EXPERIMENT 1A: NEEDLE-IN-A-HAYSTACK (VERSION 3 - PATH A FIX)")
    print("=" * 80)
    print("PATH A: Query-Needle Affinity")
    print("  ✓ Structured embeddings (filler clustered, needle distinct)")
    print("  ✓ Query token with HIGH affinity to needle")
    print("  ✓ Dense attention naturally highlights needle")
    print("  ✓ FRC analyzes real attention structure")
    print("=" * 80)
    print(f"Context lengths: {context_lengths}")
    print(f"Needle depths: {needle_depths}")
    print(f"Sparsity levels: {sparsity_levels}")
    print(f"Methods: {methods}")
    print(f"Samples per config: {num_samples}")
    print("=" * 80)

    total_configs = len(context_lengths) * len(needle_depths) * len(sparsity_levels) * len(methods)
    pbar = tqdm(total=total_configs, desc="Running NIAH v3")

    for ctx_len in context_lengths:
        for depth in needle_depths:
            for sparsity in sparsity_levels:
                samples = [create_niah_sample(ctx_len, depth) for _ in range(num_samples)]

                for method in methods:
                    successes = []
                    for sample in samples:
                        try:
                            success = evaluate_single_sample_v3(
                                sample, method, sparsity, model, block_size, device
                            )
                            successes.append(success)
                        except Exception as e:
                            print(f"\nError: {e}")
                            successes.append(False)

                    accuracy = sum(successes) / len(successes)

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
    save_path: str = 'niah_heatmap_v3.png'
):
    """Generate heatmap."""
    methods = list(results.keys())
    num_methods = len(methods)

    fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 5))
    if num_methods == 1:
        axes = [axes]

    for idx, method in enumerate(methods):
        matrix = np.zeros((len(needle_depths), len(context_lengths)))

        for i, depth in enumerate(needle_depths):
            for j, ctx_len in enumerate(context_lengths):
                key = (ctx_len, depth, sparsity)
                matrix[i, j] = results[method].get(key, 0.0)

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
    """Run Experiment 1A Version 3 (Path A)."""

    # Configuration
    context_lengths = [1024, 2048, 4096, 8192]
    needle_depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    sparsity_levels = [0.90, 0.95, 0.99]
    methods = ['full', 'h2o', 'cab']
    num_samples = 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Run experiment
    results = run_niah_experiment_v3(
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

    with open('niah_results_v3.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print("\nSaved results: niah_results_v3.json")

    # Generate heatmaps
    for sparsity in sparsity_levels:
        plot_niah_heatmap(
            results,
            context_lengths,
            needle_depths,
            sparsity,
            save_path=f'niah_heatmap_v3_sparsity_{int(sparsity*100)}.png'
        )

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (VERSION 3 - PATH A)")
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
    print("PATH A HYPOTHESIS TEST:")
    print("  If CAB >> H2O:")
    print("    ✓ FRC successfully identifies needle as bridge")
    print("    → Proceed to Path B (real LLM integration)")
    print("  If CAB ≈ H2O:")
    print("    ✗ Need to rethink FRC formulation")
    print("=" * 80)


if __name__ == '__main__':
    main()
