"""
Verify Attention Mechanisms
============================

Test which attention implementation is used when running:
1. Dense alone (should use SDPA)
2. Dense + CAB + H2O together (should use custom Flash Attention)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*60)
print("Attention Mechanism Verification")
print("="*60)

# Test 1: Dense alone (should use SDPA)
print("\n1. Testing DENSE ALONE (should use SDPA)...")
print("-" * 60)

model_kwargs = {
    'torch_dtype': torch.float16,
    'device_map': 'cuda',
    'trust_remote_code': True,
}

# Try SDPA
try:
    import torch.nn.functional as F
    if hasattr(F, 'scaled_dot_product_attention'):
        model_kwargs['attn_implementation'] = 'sdpa'
        print("✓ Attempting to load with SDPA...")
    else:
        print("✗ SDPA not available")
except:
    print("✗ SDPA check failed")

model_dense = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    **model_kwargs
)

# Check what was actually loaded
first_layer = model_dense.model.layers[0].self_attn
print(f"   Model class: {model_dense.__class__.__name__}")
print(f"   Attention class: {first_layer.__class__.__name__}")
print(f"   Has forward method: {hasattr(first_layer, 'forward')}")

# Check if it's using flash attention or sdpa
if hasattr(model_dense.config, '_attn_implementation'):
    print(f"   Config attn_implementation: {model_dense.config._attn_implementation}")
else:
    print(f"   Config attn_implementation: (not set)")

del model_dense
torch.cuda.empty_cache()

# Test 2: With Custom Flash Attention (for CAB/H2O)
print("\n2. Testing WITH CUSTOM FLASH ATTENTION (for CAB/H2O)...")
print("-" * 60)

model_kwargs_eager = {
    'torch_dtype': torch.float16,
    'device_map': 'cuda',
    'trust_remote_code': True,
    'attn_implementation': 'eager',
}

print("✓ Loading with eager attention...")
model_flash = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    **model_kwargs_eager
)

print(f"   Model class: {model_flash.__class__.__name__}")
first_layer_before = model_flash.model.layers[0].self_attn
print(f"   Attention class (before patch): {first_layer_before.__class__.__name__}")

# Patch with custom Flash Attention
try:
    from cab_attention.kernels.flash_attention_accumulate import replace_attention_with_flash
    print("✓ Patching with custom Flash Attention...")
    model_flash = replace_attention_with_flash(model_flash)

    first_layer_after = model_flash.model.layers[0].self_attn
    print(f"   Attention class (after patch): {first_layer_after.__class__.__name__}")
    print(f"   Has cumulative_scores: {hasattr(first_layer_after, 'cumulative_scores')}")

    # Check if Flash Attention forward is being used
    if hasattr(first_layer_after, 'forward'):
        import inspect
        source = inspect.getsource(first_layer_after.forward)
        if 'flash_attention_forward_with_cumulative_scores' in source:
            print("   ✓ Using custom Flash Attention forward!")
        else:
            print("   ? Forward method source unknown")

except ImportError as e:
    print(f"   ✗ Flash Attention import failed: {e}")
except Exception as e:
    print(f"   ✗ Patching failed: {e}")

# Test 3: Verify generation works with both
print("\n3. Testing generation...")
print("-" * 60)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
prompt = "What is 2+2?"
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

print("   Generating with custom Flash Attention...")
with torch.no_grad():
    outputs = model_flash.generate(**inputs, max_new_tokens=10, do_sample=False)
generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"   Generated: '{generated}'")

# Check cumulative scores were accumulated
if hasattr(model_flash.model.layers[0].self_attn, 'cumulative_scores'):
    scores = model_flash.model.layers[0].self_attn.cumulative_scores
    if scores is not None:
        print(f"   ✓ Cumulative scores accumulated: {scores.shape}")
    else:
        print(f"   ✗ Cumulative scores are None")
else:
    print(f"   ✗ No cumulative_scores attribute")

print("\n" + "="*60)
print("Summary:")
print("="*60)
print("When running DENSE ALONE:")
print("  → Uses SDPA (fast, no cumulative scores)")
print("\nWhen running DENSE + CAB + H2O TOGETHER:")
print("  → All use custom Flash Attention (with cumulative scores)")
print("  → CAB/H2O use cumulative scores for eviction")
print("  → Dense also gets custom Flash Attention (slightly different from pure SDPA)")
print("\nThis is the current implementation behavior.")
print("="*60)
