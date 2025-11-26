"""
Pre-flight test for GPT-2 NIAH experiment.

This script validates:
1. Dependencies are installed
2. GPT-2 can be loaded
3. Attention extraction works
4. Needle detection works
5. Sparse masks can be generated

Run this BEFORE the full experiment to catch any issues.
Expected runtime: ~10 seconds
"""

import sys
sys.path.insert(0, '..')

import torch

def test_imports():
    """Test that all required libraries are available."""
    print("Testing imports...")

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("  ✓ transformers")
    except ImportError:
        print("  ✗ transformers not found")
        print("  → Install with: pip install transformers")
        return False

    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy not found")
        return False

    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib not found")
        return False

    return True


def test_gpt2_loading():
    """Test that GPT-2 can be loaded."""
    print("\nTesting GPT-2 loading...")

    try:
        from transformers import GPT2Model, GPT2Tokenizer

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        print("  Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print(f"  ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

        print("  Loading model...")
        # Use 'eager' attention to enable attention output
        model = GPT2Model.from_pretrained("gpt2", attn_implementation='eager').to(device)
        model.eval()
        print(f"  ✓ Model loaded")
        print(f"    - Layers: {len(model.h)}")
        print(f"    - Heads: {model.config.n_head}")
        print(f"    - Hidden dim: {model.config.n_embd}")

        return True, tokenizer, model, device

    except Exception as e:
        print(f"  ✗ Error loading GPT-2: {e}")
        return False, None, None, None


def test_attention_extraction(tokenizer, model, device):
    """Test attention extraction from GPT-2."""
    print("\nTesting attention extraction...")

    try:
        # Create test input
        text = "The secret key is 12345. What is the key?"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"  Test text: '{text}'")
        print(f"  Tokens: {token_ids} (length: {len(token_ids)})")

        # Forward pass
        input_ids = torch.tensor([token_ids], device=device)

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True, use_cache=False)

        # Check if attentions were returned
        if outputs.attentions is None:
            print(f"  ✗ No attentions returned (outputs.attentions is None)")
            return False, None

        print(f"  ✓ Attentions tuple length: {len(outputs.attentions)}")

        # Check each layer
        for i, attn in enumerate(outputs.attentions):
            if attn is None:
                print(f"    Layer {i}: None")
            else:
                print(f"    Layer {i}: {attn.shape}")

        # Extract attention from layer 6
        attention = outputs.attentions[6]  # [1, num_heads, N, N]

        if attention is None:
            print(f"  ✗ Layer 6 attention is None!")
            return False, None

        print(f"  ✓ Attention extracted from layer 6")
        print(f"    - Shape: {attention.shape}")
        print(f"    - Heads: {attention.shape[1]}")
        print(f"    - Sequence length: {attention.shape[2]}")

        # Average across heads
        attention_avg = attention.mean(dim=1)[0]  # [N, N]
        print(f"  ✓ Averaged across heads: {attention_avg.shape}")
        print(f"    - Min: {attention_avg.min():.4f}")
        print(f"    - Max: {attention_avg.max():.4f}")
        print(f"    - Mean: {attention_avg.mean():.4f}")

        return True, attention_avg

    except Exception as e:
        print(f"  ✗ Error extracting attention: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_needle_detection(tokenizer):
    """Test needle position detection."""
    print("\nTesting needle detection...")

    try:
        # Create sample with needle
        filler_before = "The sky is blue. Water flows. "
        needle = "The secret key is 47291."
        filler_after = " Birds fly south. Rain falls."

        context = filler_before + needle + filler_after
        print(f"  Context: '{context}'")

        # Tokenize
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        needle_tokens = tokenizer.encode(needle, add_special_tokens=False)

        print(f"  Context tokens: {len(context_tokens)}")
        print(f"  Needle tokens: {len(needle_tokens)}")
        print(f"  Needle token IDs: {needle_tokens}")

        # Find needle
        needle_start = None
        for i in range(len(context_tokens) - len(needle_tokens) + 1):
            if context_tokens[i:i+len(needle_tokens)] == needle_tokens:
                needle_start = i
                break

        if needle_start is not None:
            print(f"  ✓ Needle found at position {needle_start}")
            print(f"    Positions: {list(range(needle_start, needle_start + len(needle_tokens)))}")
            return True
        else:
            print(f"  ✗ Needle not found in context")
            print(f"    Context tokens: {context_tokens}")
            print(f"    Needle tokens: {needle_tokens}")
            return False

    except Exception as e:
        print(f"  ✗ Error detecting needle: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sparse_masks(attention):
    """Test sparse mask generation."""
    print("\nTesting sparse mask generation...")

    try:
        N = attention.shape[0]
        device = attention.device

        # Test H2O mask
        print("  Testing H2O mask...")
        from exp1a_niah_passkey_gpt2 import apply_h2o_mask
        h2o_mask = apply_h2o_mask(attention, sparsity=0.9, block_size=8)
        h2o_sparsity = 1.0 - (h2o_mask.sum().item() / (N * N))
        print(f"    ✓ H2O mask generated")
        print(f"      - Target sparsity: 90%")
        print(f"      - Actual sparsity: {h2o_sparsity:.1%}")
        print(f"      - Kept elements: {h2o_mask.sum().item()} / {N*N}")

        # Test CAB mask
        print("  Testing CAB mask...")
        from exp1a_niah_passkey_gpt2 import apply_cab_mask
        cab_mask = apply_cab_mask(attention, sparsity=0.9, block_size=8, lambda_redundancy=0.5)
        cab_sparsity = 1.0 - (cab_mask.sum().item() / (N * N))
        print(f"    ✓ CAB mask generated")
        print(f"      - Target sparsity: 90%")
        print(f"      - Actual sparsity: {cab_sparsity:.1%}")
        print(f"      - Kept elements: {cab_mask.sum().item()} / {N*N}")

        return True

    except Exception as e:
        print(f"  ✗ Error generating masks: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all pre-flight tests."""
    print("=" * 80)
    print("GPT-2 NIAH EXPERIMENT - PRE-FLIGHT VALIDATION")
    print("=" * 80)

    # Test 1: Imports
    if not test_imports():
        print("\n✗ FAILED: Missing dependencies")
        return False

    # Test 2: GPT-2 Loading
    success, tokenizer, model, device = test_gpt2_loading()
    if not success:
        print("\n✗ FAILED: Could not load GPT-2")
        return False

    # Test 3: Attention Extraction
    success, attention = test_attention_extraction(tokenizer, model, device)
    if not success:
        print("\n✗ FAILED: Could not extract attention")
        return False

    # Test 4: Needle Detection
    if not test_needle_detection(tokenizer):
        print("\n✗ FAILED: Could not detect needle")
        return False

    # Test 5: Sparse Masks
    if not test_sparse_masks(attention):
        print("\n✗ FAILED: Could not generate sparse masks")
        return False

    print("\n" + "=" * 80)
    print("✓ ALL PRE-FLIGHT TESTS PASSED")
    print("=" * 80)
    print()
    print("Ready to run full experiment!")
    print("Estimated runtime: 3 minutes")
    print("Estimated memory: 6GB peak")
    print()
    print("Run with:")
    print("  python3 exp1a_niah_passkey_gpt2.py")
    print("=" * 80)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
