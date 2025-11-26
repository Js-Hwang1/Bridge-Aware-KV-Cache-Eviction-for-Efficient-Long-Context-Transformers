"""Debug script to understand why needle detection is failing."""

import sys
sys.path.insert(0, '..')

import torch
from transformers import GPT2Model, GPT2Tokenizer
import random

# Initialize
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Test 1: Simple needle detection
print("=" * 80)
print("TEST 1: Basic Needle Tokenization")
print("=" * 80)

filler = "The sky is blue. "
needle = "The secret key is 47291."
context = filler + needle + " " + filler

print(f"Context: '{context}'")
print()

# Tokenize
context_tokens = tokenizer.encode(context, add_special_tokens=False)
needle_tokens = tokenizer.encode(needle, add_special_tokens=False)

print(f"Context tokens ({len(context_tokens)}): {context_tokens}")
print(f"Needle tokens ({len(needle_tokens)}): {needle_tokens}")
print()

# Find needle
needle_found = False
needle_start = None
for i in range(len(context_tokens) - len(needle_tokens) + 1):
    if context_tokens[i:i+len(needle_tokens)] == needle_tokens:
        needle_found = True
        needle_start = i
        break

print(f"Needle found: {needle_found}")
if needle_found:
    print(f"Needle start position: {needle_start}")
    print(f"Needle positions: {list(range(needle_start, needle_start + len(needle_tokens)))}")
else:
    print("ERROR: Needle not found!")
    # Try to find partial match
    print("\nSearching for number tokens...")
    number = "47291"
    number_tokens = tokenizer.encode(number, add_special_tokens=False)
    print(f"Number tokens: {number_tokens}")

    for i in range(len(context_tokens)):
        if context_tokens[i] in number_tokens:
            print(f"  Found number token {context_tokens[i]} at position {i}")

print()
print("=" * 80)
print("TEST 2: Full NIAH Sample Generation")
print("=" * 80)

# Generate a real NIAH sample
def generate_filler_text(target_tokens):
    sentences = [
        "The sky is blue and the grass is green.",
        "Water flows down the river to the sea.",
        "Birds fly south for the winter months.",
    ]
    num_sentences = (target_tokens // 12) + 1
    return " ".join([random.choice(sentences) for _ in range(num_sentences)])

passkey = "47291"
needle_text = f"The secret key is {passkey}."
context_length = 512
needle_depth = 0.5

# Tokenize needle first
needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
needle_length = len(needle_tokens)

print(f"Target context length: {context_length}")
print(f"Needle: '{needle_text}'")
print(f"Needle tokens: {needle_tokens} (length: {needle_length})")
print()

# Generate filler
filler_tokens = context_length - needle_length
needle_position = int(filler_tokens * needle_depth)

filler_before = generate_filler_text(needle_position)
filler_after = generate_filler_text(filler_tokens - needle_position)

# Construct context
context = f"{filler_before} {needle_text} {filler_after}"
context_token_ids = tokenizer.encode(context, add_special_tokens=False)

print(f"Actual context length: {len(context_token_ids)}")
print()

# Find needle in full context
needle_start = None
for i in range(len(context_token_ids) - needle_length + 1):
    if context_token_ids[i:i+needle_length] == needle_tokens:
        needle_start = i
        break

if needle_start is not None:
    print(f"✓ Needle found at position {needle_start}")
    print(f"  Needle positions: {list(range(needle_start, needle_start + needle_length))}")
    print(f"  Needle depth in tokens: {needle_start / len(context_token_ids):.2%}")
else:
    print("✗ Needle NOT found!")
    print("\nDEBUGGING:")
    print(f"  Looking for exact sequence: {needle_tokens}")
    print(f"  In context of length: {len(context_token_ids)}")

    # Try to find passkey tokens
    passkey_tokens = tokenizer.encode(passkey, add_special_tokens=False)
    print(f"\n  Passkey tokens: {passkey_tokens}")
    for i in range(len(context_token_ids)):
        for pt in passkey_tokens:
            if context_token_ids[i] == pt:
                print(f"    Found passkey token {pt} at position {i}")

print()
print("=" * 80)
print("TEST 3: Attention Mask Evaluation Logic")
print("=" * 80)

# Simulate attention mask
N = 20
mask = torch.ones(N, N, dtype=torch.bool)

# Simulate needle at positions 10-12
needle_positions = [10, 11, 12]
query_position = N - 1  # Last position

print(f"Sequence length: {N}")
print(f"Needle positions: {needle_positions}")
print(f"Query position: {query_position}")
print()

# Check if query attends to needle
query_attention = mask[query_position, :]
success = False
for pos in needle_positions:
    if pos < N and query_attention[pos]:
        success = True
        break

print(f"Query attends to needle: {success}")
print(f"Query attention to needle positions: {[query_attention[p].item() for p in needle_positions]}")

print()
print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print()
print("The issue is likely:")
print("1. BPE tokenization makes exact needle matching difficult")
print("2. Spaces and punctuation affect token boundaries")
print("3. Need more robust needle detection (e.g., search for passkey only)")
print()
print("FIX: Search for the 5-digit passkey tokens instead of full needle phrase")
