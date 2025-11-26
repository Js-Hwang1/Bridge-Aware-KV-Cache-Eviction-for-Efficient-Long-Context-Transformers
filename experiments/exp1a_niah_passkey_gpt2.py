"""
Experiment 1A: Needle-in-a-Haystack with Real LLM (GPT-2)

PATH B: Real LLM Integration
- Uses pretrained GPT-2 where embeddings capture real semantics
- Extracts actual attention patterns from forward passes
- Tests if CAB preserves low-magnitude but structurally critical tokens better than H2O

This is the scientifically rigorous test of the FRC hypothesis.
"""

import sys
sys.path.insert(0, '..')

import torch
import torch.nn.functional as F
import numpy as np
import random
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers for GPT-2
try:
    from transformers import GPT2Model, GPT2Tokenizer
except ImportError:
    print("ERROR: transformers library not installed")
    print("Install with: pip install transformers")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    # Model settings
    model_name: str = "gpt2"  # 124M params, 12 layers, 1024 max context
    target_layer: int = 6  # Middle layer for attention analysis

    # Context settings
    context_lengths: List[int] = None  # Will be set to [512, 1024]
    needle_depths: List[float] = None  # Will be set to [0.1, 0.25, 0.5, 0.75, 0.9]
    sparsity_levels: List[float] = None  # Will be set to [0.90, 0.95, 0.99]

    # Evaluation settings
    num_samples: int = 5
    block_size: int = 64  # Block size for block-sparse attention

    # Method settings
    methods: List[str] = None  # Will be set to ['full', 'h2o', 'cab']
    lambda_redundancy: float = 0.5  # FRC redundancy weight

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.context_lengths is None:
            self.context_lengths = [512, 1024]  # Conservative start
        if self.needle_depths is None:
            self.needle_depths = [0.1, 0.25, 0.5, 0.75, 0.9]
        if self.sparsity_levels is None:
            self.sparsity_levels = [0.90, 0.95, 0.99]
        if self.methods is None:
            self.methods = ['full', 'h2o', 'cab']


# ============================================================================
# GPT-2 Needle Dataset
# ============================================================================

class GPT2NeedleDataset:
    """
    Generates NIAH samples for GPT-2.

    Key design decisions:
    1. Use simple repeated filler text (models redundancy naturally)
    2. Needle = 5-digit number (low-frequency token)
    3. Track needle positions in BPE token space
    """

    def __init__(self, tokenizer: GPT2Tokenizer):
        self.tokenizer = tokenizer

        # Filler sentences (will be repeated)
        self.filler_sentences = [
            "The sky is blue and the grass is green.",
            "Water flows down the river to the sea.",
            "Birds fly south for the winter months.",
            "The sun rises in the east every morning.",
            "Mountains tower over the valleys below.",
            "Forests are home to many wild animals.",
            "Rain falls gently on the rooftop at night.",
            "Stars twinkle brightly in the dark sky.",
            "Flowers bloom beautifully in the spring.",
            "The ocean waves crash against the shore.",
        ]

    def generate_passkey(self) -> str:
        """Generate a random 5-digit passkey."""
        return f"{random.randint(10000, 99999)}"

    def generate_filler_text(self, target_tokens: int) -> str:
        """
        Generate filler text to approximately reach target token count.

        Args:
            target_tokens: Approximate number of tokens desired

        Returns:
            Filler text string
        """
        # Each sentence is roughly 10-15 tokens
        num_sentences = (target_tokens // 12) + 1
        sentences = [random.choice(self.filler_sentences) for _ in range(num_sentences)]
        return " ".join(sentences)

    def create_sample(
        self,
        context_length: int,
        needle_depth: float,
        passkey: Optional[str] = None
    ) -> Dict:
        """
        Create a single NIAH sample.

        Args:
            context_length: Target context length in tokens
            needle_depth: Position of needle (0.0 = start, 1.0 = end)
            passkey: Optional specific passkey (for reproducibility)

        Returns:
            Sample dict with context, needle info, and token positions
        """
        if passkey is None:
            passkey = self.generate_passkey()

        # Create needle text
        needle_text = f"The secret key is {passkey}."

        # Tokenize needle to get exact length
        needle_tokens = self.tokenizer.encode(needle_text, add_special_tokens=False)
        needle_length = len(needle_tokens)

        # Calculate filler lengths
        filler_tokens = context_length - needle_length
        needle_position = int(filler_tokens * needle_depth)

        # Generate filler
        filler_before = self.generate_filler_text(needle_position)
        filler_after = self.generate_filler_text(filler_tokens - needle_position)

        # Construct full context
        context = f"{filler_before} {needle_text} {filler_after}"

        # Tokenize full context
        context_token_ids = self.tokenizer.encode(context, add_special_tokens=False)

        # Find needle positions in tokenized sequence
        needle_start_pos = None
        for i in range(len(context_token_ids) - needle_length + 1):
            if context_token_ids[i:i+needle_length] == needle_tokens:
                needle_start_pos = i
                break

        if needle_start_pos is None:
            # Fallback: search for passkey tokens only
            passkey_tokens = self.tokenizer.encode(passkey, add_special_tokens=False)
            for i in range(len(context_token_ids) - len(passkey_tokens) + 1):
                if context_token_ids[i:i+len(passkey_tokens)] == passkey_tokens:
                    needle_start_pos = i
                    needle_length = len(passkey_tokens)
                    break

        return {
            'context': context,
            'context_token_ids': context_token_ids,
            'needle_text': needle_text,
            'passkey': passkey,
            'needle_positions': list(range(needle_start_pos, needle_start_pos + needle_length)) if needle_start_pos is not None else [],
            'actual_length': len(context_token_ids),
            'target_length': context_length,
        }


# ============================================================================
# GPT-2 Attention Extractor
# ============================================================================

class GPT2AttentionExtractor:
    """
    Extracts attention matrices from GPT-2.

    Design:
    - Loads GPT-2 in eval mode
    - Runs forward pass with output_attentions=True
    - Returns attention from target layer
    - Memory efficient (no gradient computation)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        target_layer: int = 6,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.target_layer = target_layer
        self.device = device

        print(f"Loading {model_name}...")
        # Use 'eager' attention implementation to enable attention output
        self.model = GPT2Model.from_pretrained(
            model_name,
            attn_implementation='eager'
        ).to(device)
        self.model.eval()

        self.num_layers = len(self.model.h)
        self.num_heads = self.model.config.n_head
        self.hidden_dim = self.model.config.n_embd

        print(f"✓ Model loaded: {self.num_layers} layers, {self.num_heads} heads, {self.hidden_dim} dim")

        if target_layer >= self.num_layers:
            raise ValueError(f"target_layer {target_layer} >= num_layers {self.num_layers}")

    @torch.no_grad()
    def extract_attention(
        self,
        token_ids: List[int]
    ) -> torch.Tensor:
        """
        Extract attention matrix from target layer.

        Args:
            token_ids: List of token IDs

        Returns:
            attention: [N, N] attention matrix (averaged across heads)
        """
        # Convert to tensor
        input_ids = torch.tensor([token_ids], device=self.device)  # [1, N]

        # Forward pass with attention output
        outputs = self.model(
            input_ids,
            output_attentions=True,
            use_cache=False  # Don't cache key/values (saves memory)
        )

        # Extract attention from target layer
        # outputs.attentions is tuple of (layer0, layer1, ..., layer11)
        # Each layer: [batch_size, num_heads, seq_len, seq_len]
        layer_attention = outputs.attentions[self.target_layer]  # [1, num_heads, N, N]

        # Average across heads
        attention_avg = layer_attention.mean(dim=1)[0]  # [N, N]

        return attention_avg

    def get_model_info(self) -> Dict:
        """Return model information."""
        return {
            'model_name': self.model_name,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'hidden_dim': self.hidden_dim,
            'target_layer': self.target_layer,
        }


# ============================================================================
# Sparse Mask Generators
# ============================================================================

def apply_h2o_mask(
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 64
) -> torch.Tensor:
    """
    H2O: Keep top-k blocks by maximum attention magnitude.

    Args:
        attention: [N, N] dense attention matrix
        sparsity: Fraction to prune (0.9 = keep 10%)
        block_size: Block size for block-sparse masking

    Returns:
        mask: [N, N] boolean mask (True = keep)
    """
    N = attention.shape[0]
    device = attention.device

    # Blockify: compute max attention per block
    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=device)

    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = attention[i_start:i_end, j_start:j_end].max()

    # Select top-k blocks by magnitude
    k_keep = max(1, int(M * M * (1 - sparsity)))
    threshold = torch.topk(block_scores.flatten(), k_keep).values[-1]
    block_mask = block_scores >= threshold

    # Expand to token level
    token_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(M):
        for j in range(M):
            if block_mask[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask


def apply_cab_mask(
    attention: torch.Tensor,
    sparsity: float,
    block_size: int = 64,
    lambda_redundancy: float = 0.5
) -> torch.Tensor:
    """
    CAB: Keep blocks with lowest FRC (bridges in attention graph).

    FRC Formula:
        FRC[i,j] = attention[i,j] - λ * Σ_k attention[i,k] * attention[k,j]

    Low FRC = bridge (critical connector) → KEEP
    High FRC = redundant (alternative paths) → PRUNE

    Args:
        attention: [N, N] dense attention matrix
        sparsity: Fraction to prune
        block_size: Block size
        lambda_redundancy: Weight for redundancy term

    Returns:
        mask: [N, N] boolean mask
    """
    N = attention.shape[0]
    device = attention.device

    # Blockify: average attention per block
    M = (N + block_size - 1) // block_size
    block_scores = torch.zeros(M, M, device=device)

    for i in range(M):
        for j in range(M):
            i_start, i_end = i * block_size, min((i + 1) * block_size, N)
            j_start, j_end = j * block_size, min((j + 1) * block_size, N)
            block_scores[i, j] = attention[i_start:i_end, j_start:j_end].mean()

    # Compute FRC on block graph
    # Direct connections
    direct = block_scores

    # Redundancy: 2-hop paths (triangles)
    # redundancy[i,j] = Σ_k block_scores[i,k] * block_scores[k,j]
    redundancy = torch.matmul(block_scores, block_scores)

    # FRC score (lower = more critical)
    frc_scores = direct - lambda_redundancy * redundancy

    # Keep blocks with LOWEST FRC (bridges)
    k_keep = max(1, int(M * M * (1 - sparsity)))
    threshold = torch.topk(frc_scores.flatten(), k_keep, largest=False).values[-1]
    block_mask = frc_scores <= threshold

    # Expand to token level
    token_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(M):
        for j in range(M):
            if block_mask[i, j]:
                i_start, i_end = i * block_size, min((i + 1) * block_size, N)
                j_start, j_end = j * block_size, min((j + 1) * block_size, N)
                token_mask[i_start:i_end, j_start:j_end] = True

    return token_mask


# ============================================================================
# Needle Evaluator
# ============================================================================

def evaluate_needle_retrieval(
    attention_mask: torch.Tensor,
    needle_positions: List[int],
    query_position: Optional[int] = None
) -> bool:
    """
    Check if needle tokens are preserved in sparse attention.

    Success criteria: Query position attends to at least one needle token.

    Args:
        attention_mask: [N, N] boolean mask (True = attended)
        needle_positions: List of needle token indices
        query_position: Query token index (defaults to last position)

    Returns:
        success: True if any needle position is attended
    """
    N = attention_mask.shape[0]

    if query_position is None:
        query_position = N - 1  # Last position is the query

    if not needle_positions:
        return False

    # Check if query attends to any needle position
    query_attention = attention_mask[query_position, :]

    for pos in needle_positions:
        if pos < N and query_attention[pos]:
            return True

    return False


# ============================================================================
# Experiment Runner
# ============================================================================

def run_single_sample(
    sample: Dict,
    extractor: GPT2AttentionExtractor,
    method: str,
    sparsity: float,
    config: ExperimentConfig
) -> Tuple[bool, Dict]:
    """
    Evaluate a single NIAH sample.

    Args:
        sample: Sample dict from GPT2NeedleDataset
        extractor: GPT2AttentionExtractor instance
        method: 'full', 'h2o', or 'cab'
        sparsity: Sparsity level
        config: Experiment configuration

    Returns:
        success: True if retrieval succeeded
        info: Additional info for debugging
    """
    token_ids = sample['context_token_ids']
    needle_positions = sample['needle_positions']

    # Extract dense attention from GPT-2
    dense_attention = extractor.extract_attention(token_ids)
    N = dense_attention.shape[0]

    # Apply sparse masking based on method
    if method == 'full':
        mask = torch.ones(N, N, dtype=torch.bool, device=dense_attention.device)
    elif method == 'h2o':
        mask = apply_h2o_mask(dense_attention, sparsity, config.block_size)
    elif method == 'cab':
        mask = apply_cab_mask(
            dense_attention,
            sparsity,
            config.block_size,
            config.lambda_redundancy
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Evaluate retrieval
    success = evaluate_needle_retrieval(mask, needle_positions)

    # Compute some statistics for debugging
    info = {
        'actual_length': N,
        'needle_positions': needle_positions,
        'num_needle_tokens': len(needle_positions),
        'sparsity_actual': 1.0 - (mask.sum().item() / (N * N)),
    }

    return success, info


def run_experiment(config: ExperimentConfig) -> Dict:
    """
    Run full NIAH experiment with GPT-2.

    Returns:
        results: Nested dict of {method: {(ctx_len, depth, sparsity): accuracy}}
    """
    # Initialize components
    print("=" * 80)
    print("EXPERIMENT 1A: NEEDLE-IN-A-HAYSTACK WITH GPT-2")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Target layer: {config.target_layer}")
    print(f"Context lengths: {config.context_lengths}")
    print(f"Needle depths: {config.needle_depths}")
    print(f"Sparsity levels: {config.sparsity_levels}")
    print(f"Methods: {config.methods}")
    print(f"Samples per config: {config.num_samples}")
    print(f"Device: {config.device}")
    print("=" * 80)

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    dataset = GPT2NeedleDataset(tokenizer)
    extractor = GPT2AttentionExtractor(
        config.model_name,
        config.target_layer,
        config.device
    )

    # Results storage
    results = {method: {} for method in config.methods}

    # Total configurations
    total_configs = (
        len(config.context_lengths) *
        len(config.needle_depths) *
        len(config.sparsity_levels) *
        len(config.methods)
    )

    pbar = tqdm(total=total_configs, desc="Running NIAH (GPT-2)")

    # Main experiment loop
    for ctx_len in config.context_lengths:
        for depth in config.needle_depths:
            # Generate samples for this configuration
            samples = [
                dataset.create_sample(ctx_len, depth)
                for _ in range(config.num_samples)
            ]

            for sparsity in config.sparsity_levels:
                for method in config.methods:
                    successes = []

                    for sample in samples:
                        try:
                            success, info = run_single_sample(
                                sample, extractor, method, sparsity, config
                            )
                            successes.append(success)
                        except Exception as e:
                            print(f"\nError: {e}")
                            import traceback
                            traceback.print_exc()
                            successes.append(False)

                    # Compute accuracy
                    accuracy = sum(successes) / len(successes)

                    # Store results
                    key = (ctx_len, depth, sparsity)
                    results[method][key] = accuracy

                    # Update progress
                    pbar.set_postfix({
                        'method': method,
                        'N': ctx_len,
                        'depth': f'{depth:.2f}',
                        'sparsity': f'{sparsity:.0%}',
                        'acc': f'{accuracy:.2f}'
                    })
                    pbar.update(1)

    pbar.close()

    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_heatmap(
    results: Dict,
    context_lengths: List[int],
    needle_depths: List[float],
    sparsity: float,
    save_path: str = 'niah_heatmap_gpt2.png'
):
    """Generate comparison heatmap."""
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
            xticklabels=[f'{n}' for n in context_lengths],
            yticklabels=[f'{int(d*100)}%' for d in needle_depths],
            cmap='RdYlGn',
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Accuracy'}
        )
        axes[idx].set_title(
            f'{method.upper()} (Sparsity={sparsity:.0%})',
            fontsize=14,
            fontweight='bold'
        )
        axes[idx].set_xlabel('Context Length', fontsize=12)
        axes[idx].set_ylabel('Needle Depth', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved heatmap: {save_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    """Main experiment entry point."""

    # Configuration
    config = ExperimentConfig()

    print("\n" + "=" * 80)
    print("SCIENTIFIC HYPOTHESIS TEST")
    print("=" * 80)
    print("Research Question:")
    print("  Can geometric (curvature-based) sparse attention preserve")
    print("  low-magnitude but structurally critical tokens better than")
    print("  magnitude-based methods?")
    print()
    print("Expected Results:")
    print("  Full Attention: 100% (upper bound)")
    print("  H2O (Magnitude): <50% (drops low-magnitude needles)")
    print("  CAB (Curvature): >90% (preserves bridges)")
    print("=" * 80)
    print()

    # Run experiment
    results = run_experiment(config)

    # Save results
    results_serializable = {
        method: {str(k): v for k, v in method_results.items()}
        for method, method_results in results.items()
    }

    with open('niah_results_gpt2.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print("\n✓ Saved results: niah_results_gpt2.json")

    # Generate heatmaps
    for sparsity in config.sparsity_levels:
        plot_heatmap(
            results,
            config.context_lengths,
            config.needle_depths,
            sparsity,
            save_path=f'niah_heatmap_gpt2_sparsity_{int(sparsity*100)}.png'
        )

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (GPT-2)")
    print("=" * 80)

    for sparsity in config.sparsity_levels:
        print(f"\nSparsity: {sparsity:.0%}")
        print(f"{'Method':<10} | {'Avg Accuracy':<15} | {'Min Accuracy':<15}")
        print("-" * 50)

        for method in config.methods:
            accs = [v for k, v in results[method].items() if k[2] == sparsity]
            if accs:
                avg_acc = np.mean(accs)
                min_acc = np.min(accs)
                print(f"{method.upper():<10} | {avg_acc:>14.2%} | {min_acc:>14.2%}")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  If CAB >> H2O: ✓ FRC preserves critical bridges")
    print("                 → ICML contribution validated")
    print("                 → Scale to longer contexts (8k, 16k, 32k)")
    print()
    print("  If CAB ≈ H2O:  ✗ Need to investigate FRC formulation")
    print("                 → Adjust λ hyperparameter")
    print("                 → Try different layers")
    print("=" * 80)


if __name__ == '__main__':
    main()
