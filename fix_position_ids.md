# Fix: Position IDs Corruption in CAB/H2O

## Root Cause

When evicting tokens from KV cache, position IDs are not being tracked. This causes RoPE (Rotary Position Embeddings) to be misaligned, resulting in corrupted generation.

### Example:
1. Prefill: 1000 tokens → KV cache has 1000 entries at positions [0...999]
2. Evict to 100 tokens (90% sparsity) → Keep indices [900...999] (recent 100)
3. Generate token 1001:
   - **Current (buggy)**: No position_ids passed → model assumes position = cache_len = 100
   - **Correct**: Should pass position_ids = 1000

## The Fix

### Modify `_sparse_generate`:

1. Track `current_position` starting from initial sequence length
2. Pass `position_ids` parameter during generation
3. Increment `current_position` after each token (don't reset after eviction)

### Code Changes:

```python
# In _sparse_generate, after initial prefill (line 580):
current_position = inputs['input_ids'].shape[1]  # Track actual position

# In generation loop (line 624):
outputs = self.model(
    input_ids=next_token,
    past_key_values=past_key_values,
    position_ids=torch.tensor([[current_position]], device=device),  # ADD THIS
    use_cache=True,
    return_dict=True,
    output_attentions=need_attention,
)

# After each token (line 612):
current_position += 1  # Increment regardless of eviction
```

### Why This Works:

- RoPE embeddings are position-dependent
- Even though KV cache shrinks, new tokens must know their TRUE position in the full sequence
- Position IDs ensure correct attention computation with evicted cache

## Testing:

Run with fix:
```bash
python -m experiments.longbench_qa.driver \
  --model Qwen/Qwen2.5-7B-Instruct \
  --methods cab h2o \
  --datasets hotpotqa \
  --sparsity 0.9 \
  --max-samples 5
```

Expected: Clean outputs instead of gibberish.
