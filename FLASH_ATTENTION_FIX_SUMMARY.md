# Flash Attention Implementation - Fix Summary

## Problem Overview

CAB and H2O were producing gibberish outputs with F1=0% when using custom Flash Attention for KV cache eviction.

## Root Causes Identified & Fixed

### ✅ Issue 1: Softmax Padding Mask (FIXED)
**Problem**: When sequence length < BLOCK_N (128), the kernel loaded zeros for masked positions, creating attention scores of 0 instead of -inf.

**Fix** ([cab_attention/kernels/flash_attention_accumulate.py:103-104](cab_attention/kernels/flash_attention_accumulate.py#L103-L104)):
```python
valid_mask = (q_offs[:, None] < N_q) & (k_offs[None, :] < N_k)
qk = tl.where(valid_mask, qk, float('-inf'))
```

**Impact**: Kernel now passes all tests (max diff < 0.001 vs SDPA)

### ✅ Issue 2: NaN in Cumulative Scores (FIXED)
**Problem**: When computing `exp(-inf - (-inf))` for fully masked rows → `exp(nan) = nan`. All cumulative scores were NaN, causing random eviction.

**Symptoms**:
- Before: Range [nan, nan], Sum: nan, Non-zero: 0/532
- Outputs: "medal medal medal..." (repetitive gibberish)

**Fix** ([cab_attention/kernels/flash_attention_accumulate.py:112-123](cab_attention/kernels/flash_attention_accumulate.py#L112-L123)):
```python
m_ij = tl.maximum(m_i, tl.max(qk, axis=1))

# Handle numerical stability for masked rows
all_masked = (m_ij == float('-inf'))
m_ij = tl.where(all_masked, 0.0, m_ij)

p = tl.exp(qk - m_ij[:, None])

# Zero out probabilities for fully masked rows
p = tl.where(all_masked[:, None], 0.0, p)
```

**Impact**:
- After: Range [0.0000, 16.4796], Sum: 2058.1914, Non-zero: 532/532 ✅
- Outputs: Coherent English sentences ✅

### ✅ Issue 3: Cumulative Score Contamination (FIXED)
**Problem**: Scores accumulated across samples without reset, causing stale scores to affect new samples.

**Fix** ([experiments/longbench_qa/runner.py:480-491](experiments/longbench_qa/runner.py#L480-L491)):
```python
def _reset_flash_cumulative_scores(self) -> None:
    """Reset cumulative attention scores before each sample."""
    if not getattr(self, 'use_flash_attention', False):
        return

    for layer in self.model.model.layers:
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_flash_cumulative_scores'):
            layer.self_attn._flash_cumulative_scores = None
```

Called before each sample ([runner.py:642-646](experiments/longbench_qa/runner.py#L642-L646)).

## Current Status

### ✅ Working:
1. Flash Attention kernel produces correct outputs (matches SDPA)
2. Cumulative scores are valid numbers (no NaN)
3. Outputs are coherent English sentences
4. Model doesn't crash or produce gibberish repetition

### ❌ Still Issues:
1. **F1 Score = 0%**: Answers are coherent but wrong
   - Example: "Mawson...electoral district...wine region...55 south of Adelaide"
   - Expected: "McLaren Vale"
   - Issue: Wrong information retained after eviction

2. **Possible Causes**:
   - Eviction logic not prioritizing question-relevant tokens
   - Position IDs not being tracked correctly during generation (see [POSITION_IDS_FIX_SUMMARY.md](POSITION_IDS_FIX_SUMMARY.md))
   - CAB's three-component policy (local + bridges + importance) not working correctly with Flash Attention

## Test Results

### Cumulative Scores Test (debug_cumulative_scores.py)
```
✓ Found scores from 28 layers
  Layer 0: Range [0.0000, 16.4796], Sum: 2058.1914, Non-zero: 532/532
  Aggregated: Range [782.2731, 12063.4873], Sum: 36829.0781
```

### CAB Test (3 samples, 90% sparsity)
```
Sample 1:
  Pred: "Mawson [mɛːzn] is an electoral district that includes the wine region around which town 55 south of Adelaide."
  Ref:  "McLaren Vale"
  F1: 0.000

Sample 2:
  Pred: "Re Kar Kar won the 1922 Olympic Games in the men's scull, two time World Champion..."
  Ref:  "Oberschleißheim"
  F1: 0.000
```

## Files Modified

1. **[cab_attention/kernels/flash_attention_accumulate.py](cab_attention/kernels/flash_attention_accumulate.py)**
   - Lines 103-104: Added padding mask for softmax
   - Lines 112-123: Added NaN handling for fully masked rows

2. **[experiments/longbench_qa/runner.py](experiments/longbench_qa/runner.py)**
   - Lines 288-303: Enabled Flash Attention patching for CAB/H2O
   - Lines 480-491: Added cumulative score reset function
   - Lines 642-646: Reset scores before each sample

## Commits

- `214e433`: Fix: Mask invalid positions in Flash Attention softmax
- `1eea57b`: Fix: Handle NaN in Flash Attention cumulative scores
- `0a50283`: Fix: Reset Flash Attention cumulative scores before each sample
- `cf43d8a`: Enable custom Flash Attention for CAB/H2O

## Next Steps

1. **Debug Eviction Logic**: Investigate why correct information is being evicted
   - Add logging to show which tokens have highest importance scores
   - Compare with ground truth to see if answer tokens are being kept

2. **Verify Position IDs**: Check if position IDs are being passed correctly during generation with Flash Attention (similar to fix in [POSITION_IDS_FIX_SUMMARY.md](POSITION_IDS_FIX_SUMMARY.md))

3. **Baseline Comparison**: Test Dense (no eviction) to verify F1 > 0% before debugging eviction

## Debug Scripts

- [debug_flash_kernel.py](debug_flash_kernel.py): Compare Flash Attention vs SDPA outputs
- [debug_cumulative_scores.py](debug_cumulative_scores.py): Verify cumulative scores are valid
- [test_flash_generation.py](test_flash_generation.py): Test end-to-end generation

## Clean Up

Test files to remove after debugging:
- debug_flash_attention.py
- debug_flash_kernel.py
- debug_cumulative_scores.py
- test_flash_generation.py
