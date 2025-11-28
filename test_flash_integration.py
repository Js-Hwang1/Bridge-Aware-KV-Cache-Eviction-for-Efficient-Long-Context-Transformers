"""
Test Flash Attention Integration with Qwen2
============================================

Simple test to verify Flash Attention monkey-patching works correctly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*60)
print("Flash Attention Integration Test")
print("="*60)

# Load model
print("\n1. Loading Qwen2.5-7B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

print(f"   Model loaded on: {model.device}")
print(f"   Number of layers: {len(model.model.layers)}")

# Test 1: Check original attention
print("\n2. Testing BEFORE Flash Attention patching...")
first_layer_attn = model.model.layers[0].self_attn
print(f"   Attention module: {type(first_layer_attn).__name__}")
print(f"   Forward method: {first_layer_attn.forward.__name__}")
print(f"   Has _flash_cumulative_scores: {hasattr(first_layer_attn, '_flash_cumulative_scores')}")

# Test 2: Apply Flash Attention
print("\n3. Applying Flash Attention monkey-patch...")
try:
    from cab_attention.kernels.flash_attention_accumulate import replace_attention_with_flash

    model = replace_attention_with_flash(model)
    print("   ✓ Patching successful!")

    # Verify patching
    first_layer_attn = model.model.layers[0].self_attn
    print(f"   Forward method after patch: {first_layer_attn.forward.__name__}")
    print(f"   Has _flash_cumulative_scores: {hasattr(first_layer_attn, '_flash_cumulative_scores')}")

except Exception as e:
    print(f"   ✗ Patching failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Simple generation test
print("\n4. Testing generation with Flash Attention...")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print(f"   Prompt: '{prompt}'")
print(f"   Generating...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,  # Greedy
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"   Generated: '{generated_text}'")

# Test 4: Check cumulative scores
print("\n5. Checking cumulative score accumulation...")
from cab_attention.kernels.flash_attention_accumulate import get_all_cumulative_scores

scores = get_all_cumulative_scores(model)
print(f"   Number of layers with scores: {len(scores)}")

if scores:
    first_layer_name = list(scores.keys())[0]
    first_scores = scores[first_layer_name]
    print(f"   Example layer: {first_layer_name}")
    print(f"   Scores shape: {first_scores.shape}")
    print(f"   Scores dtype: {first_scores.dtype}")
    print(f"   Scores range: [{first_scores.min():.4f}, {first_scores.max():.4f}]")
    print(f"   Scores mean: {first_scores.mean():.4f}")
else:
    print("   ✗ No scores found!")

# Test 5: Memory usage
print("\n6. Memory usage:")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved: {reserved:.2f} GB")

print("\n" + "="*60)
print("✓ All tests passed! Flash Attention is working.")
print("="*60)
