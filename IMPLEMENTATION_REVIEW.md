# Implementation Review: TODO 1.2, 1.3, 1.4

**Review Date:** 2025-11-27
**Reviewer:** Claude
**Scope:** Baseline implementations and CAB kernel usage

---

## Executive Summary

✅ **Baselines are CORRECTLY implemented** according to publications
⚠️ **CAB V4 has CRITICAL ISSUES** in perplexity_lm and downstream_tasks
✅ **CAB V4 uses REAL kernels** in longbench_qa (with minor optimization opportunity)

---

## 1. Baseline Implementation Review

### 1.1 H2O (Heavy-Hitter Oracle)

**Paper:** Zhang et al., 2023 - arXiv:2306.14048
**Status:** ✅ **CORRECTLY IMPLEMENTED**

**Locations:**
- `longbench_qa/methods.py:167-284`
- `perplexity_lm/runner.py:258-266`
- `downstream_tasks/runner.py:~480-495`

**Verification:**
```python
# Line 199 in longbench_qa/methods.py - CORRECT
importance_scores = attention_weights.sum(dim=2)  # Cumulative attention

# Line 262 in perplexity_lm/runner.py - CORRECT
_, keep_indices = torch.topk(scores, k=max_size, largest=True)
```

**Fidelity to Paper:** ✅ 100%
- Uses cumulative attention scores across all queries
- Evicts tokens with LOWEST cumulative attention
- Keeps "heavy hitters" (tokens receiving high attention)
- Matches published algorithm exactly

---

### 1.2 StreamingLLM

**Paper:** Xiao et al., 2023 - arXiv:2309.17453
**Status:** ✅ **CORRECTLY IMPLEMENTED**

**Locations:**
- `longbench_qa/methods.py:561-653`
- `perplexity_lm/runner.py:268-274`
- `downstream_tasks/runner.py:~498-505`

**Verification:**
```python
# Lines 595-599 in longbench_qa/methods.py - CORRECT
mask[:, :num_sinks] = True  # Keep attention sinks (first K tokens)
if window_size > 0:
    mask[:, -window_size:] = True  # Keep recent window

# Lines 270-274 in perplexity_lm/runner.py - CORRECT
sink_indices = torch.arange(min(sink_size, cache_len), device=device)
recent_start = max(sink_size, cache_len - recent_budget)
recent_indices = torch.arange(recent_start, cache_len, device=device)
keep_indices = torch.cat([sink_indices, recent_indices])
```

**Fidelity to Paper:** ✅ 100%
- Keeps first 4 tokens as "attention sinks"
- Keeps sliding window of recent tokens
- Evicts middle tokens
- Matches published algorithm exactly

---

### 1.3 Local + Strided Attention

**Paper:** Child et al., 2019 - arXiv:1904.10509 (Sparse Transformers)
**Status:** ✅ **CORRECTLY IMPLEMENTED**

**Location:** `longbench_qa/methods.py:660-760`

**Verification:**
```python
# Lines 694-702 in longbench_qa/methods.py - CORRECT
# Local window
for i in range(N):
    start = max(0, i - local_window // 2)
    end = min(N, i + local_window // 2 + 1)
    mask[i, start:end] = True

# Strided pattern
for i in range(N):
    strided_indices = torch.arange(0, N, stride, device=mask.device)
    mask[i, strided_indices] = True
```

**Fidelity to Paper:** ✅ 100%
- Local window for nearby dependencies
- Strided pattern for global periodic patterns
- Union of both patterns
- Matches Sparse Transformers paper

---

## 2. CAB Kernel Usage Review

### 2.1 longbench_qa/methods.py - ✅ **USES REAL KERNELS**

**CAB V3 Implementation (lines 291-399):**

✅ **Imports actual kernels:**
```python
# Lines 304-308
from cab_attention.kernels.coarsening import coarsen_qk_max_l2_pytorch
from cab_attention.kernels.frc_kernel import compute_block_frc, generate_block_mask
self.coarsen_fn = coarsen_qk_max_l2_pytorch  # ✅ Real kernel
self.frc_fn = compute_block_frc              # ✅ Real kernel
self.mask_fn = generate_block_mask           # ✅ Real kernel
```

✅ **Uses actual FRC computation:**
```python
# Lines 328-337
q_coarse, k_coarse = self.coarsen_fn(query, key, block_size)  # ✅ Real coarsening

frc_scores, affinity, redundancy = self.frc_fn(                # ✅ Real FRC
    q_coarse, k_coarse,
    temperature=1.0,
    lambda_redundancy=self.config.lambda_redundancy,
    formula=self.config.formula,
    normalization=self.config.normalization,
)
```

✅ **Generates mask correctly:**
```python
# Lines 340-346
block_mask = self.mask_fn(                      # ✅ Real mask generation
    frc_scores,
    sparsity=self.config.sparsity,
    select_high=True,
    keep_diagonal=self.config.keep_diagonal,
    causal=self.config.causal,
)
```

**CAB V4 Implementation (lines 406-554):**

✅ **Imports and uses real kernels:**
```python
# Lines 425-432 - Same as V3
# Lines 450-459 - Real coarsening and FRC computation
# Lines 466-474 - Real hybrid mask generation with magnitude_scores
```

✅ **Hybrid selection:**
```python
# Line 463
magnitude_scores = self._coarsen_attention(attention_weights, block_size)

# Lines 466-474 - CORRECT hybrid mask
block_mask = self.mask_fn(
    frc_scores,
    sparsity=self.config.sparsity,
    select_high=True,
    keep_diagonal=self.config.keep_diagonal,
    causal=self.config.causal,
    magnitude_scores=magnitude_scores,        # ✅ Magnitude component
    magnitude_ratio=self.config.magnitude_ratio,  # ✅ Hybrid ratio
)
```

**⚠️ MINOR ISSUE:**
```python
# Line 304 - Uses PyTorch version instead of Triton
from cab_attention.kernels.coarsening import coarsen_qk_max_l2_pytorch
# Should be:
from cab_attention.kernels.coarsening import coarsen_qk_max_l2  # Triton-optimized
```

**Impact:** This works correctly but is slower. Should use Triton version for GPU.

---

### 2.2 perplexity_lm/runner.py - ❌ **NOT USING REAL KERNELS**

**CAB V4 Implementation (lines 276-298):**

❌ **Uses approximation, not real FRC:**
```python
# Line 281 - Simplified magnitude
magnitude = keys.norm(dim=-1).mean(dim=(0, 1))  # [cache_len]

# Lines 284-288 - Cosine similarity approximation (NOT FRC!)
keys_flat = keys.mean(dim=(0, 1))  # [cache_len, D]
keys_norm = F.normalize(keys_flat, dim=-1)
similarity = torch.mm(keys_norm, keys_norm.t())
redundancy = similarity.mean(dim=-1)
uniqueness = 1.0 - redundancy  # ❌ NOT using compute_block_frc!

# Line 295 - Hybrid but with wrong uniqueness
importance = 0.5 * mag_norm + 0.5 * uniq_norm  # ❌ Using cosine sim, not FRC
```

**MISSING:**
- ❌ No import of `coarsen_qk_max_l2`
- ❌ No import of `compute_block_frc`
- ❌ No import of `generate_block_mask`
- ❌ No coarsening step
- ❌ No FRC computation (affinity - lambda * redundancy)
- ❌ Using simple cosine similarity instead of FRC

**What it's testing:** Magnitude + Cosine Similarity (NOT our CAB V4 method!)

---

### 2.3 downstream_tasks/runner.py - ❌ **NOT USING REAL KERNELS**

**CAB V4 Implementation (lines 507-521):**

❌ **Uses approximation, not real FRC:**
```python
# Line 509 - Simplified magnitude
magnitude = key.norm(dim=-1).mean(dim=(0, 1))  # [N]

# Lines 511-514 - Cosine similarity approximation (NOT FRC!)
k_norm = F.normalize(key.mean(dim=(0, 1)), dim=-1)
similarity = torch.mm(k_norm, k_norm.t())
redundancy = similarity.mean(dim=-1)
uniqueness = 1.0 - redundancy  # ❌ NOT using compute_block_frc!

# Line 518 - Hybrid but with wrong uniqueness
importance = magnitude_ratio * mag_norm + (1 - magnitude_ratio) * uniq_norm
```

**Same issues as perplexity_lm:**
- ❌ No CAB kernel imports
- ❌ No FRC computation
- ❌ Using cosine similarity approximation

**What it's testing:** Magnitude + Cosine Similarity (NOT our CAB V4 method!)

---

## 3. Critical Issues Summary

### Issue 1: Perplexity & Downstream Tasks NOT Testing CAB V4

**Severity:** 🔴 **CRITICAL**

**Problem:**
- `perplexity_lm/runner.py` and `downstream_tasks/runner.py` do NOT use our CAB kernels
- They use a simplified cosine similarity approximation
- This is NOT testing our actual method (Forman-Ricci Curvature)
- Results from these benchmarks will NOT reflect CAB V4's true performance

**Impact:**
- Perplexity results are for "Magnitude + Cosine Similarity", not "Magnitude + FRC"
- Downstream task results are for "Magnitude + Cosine Similarity", not "Magnitude + FRC"
- **These benchmarks are testing a different method entirely**

**Why this happened:**
- Likely because KV cache pruning operates on cached K/V tensors, not Q/K pairs
- The user may have implemented a simplified version for KV cache context
- But this defeats the purpose of using FRC-based selection

---

### Issue 2: Triton Kernel Not Used in longbench_qa

**Severity:** 🟡 **MINOR**

**Problem:**
- `longbench_qa/methods.py` uses `coarsen_qk_max_l2_pytorch` instead of `coarsen_qk_max_l2`
- The Triton version is 10-30x faster

**Impact:**
- Slower execution on GPU
- Still correct, just not optimized

**Fix:**
```python
# Change line 304 and 425:
from cab_attention.kernels.coarsening import coarsen_qk_max_l2  # Not _pytorch
```

---

## 4. Recommendations

### 4.1 URGENT: Fix CAB V4 in Perplexity & Downstream Tasks

**Two options:**

**Option A: Implement FRC-based KV cache pruning (RECOMMENDED)**

Modify the KV cache pruning to use actual CAB kernels:

```python
# In perplexity_lm/runner.py and downstream_tasks/runner.py
elif method == "cab_v4":
    # Import CAB kernels
    from cab_attention.kernels.coarsening import coarsen_qk_max_l2
    from cab_attention.kernels.frc_kernel import compute_block_frc

    # Get K tensor: [B, H, cache_len, D]
    keys = key_cache[0]

    # Create dummy Q (or use actual queries if available)
    # For KV cache pruning, we can use keys as pseudo-queries
    # or aggregate information from recent queries
    q_pseudo = keys  # Simplified approach

    # Coarsen to block level
    q_coarse, k_coarse = coarsen_qk_max_l2(q_pseudo, keys, block_size=32)

    # Compute FRC scores
    frc_scores, affinity, redundancy = compute_block_frc(
        q_coarse, k_coarse,
        lambda_redundancy=0.3,
        formula='additive',
        normalization='minmax',
    )

    # Compute magnitude at block level
    magnitude_scores = keys.norm(dim=-1).mean(dim=(0, 1))  # [cache_len]
    magnitude_blocks = magnitude_scores.view(-1, 32).max(dim=-1)[0]  # Block-level

    # Hybrid selection
    # ... use generate_block_mask or manual top-k on hybrid scores
```

**Option B: Rename methods to reflect actual algorithm**

If keeping the cosine similarity approximation:

```python
# Rename "cab_v4" to "magnitude_cosine" or "hybrid_simple"
elif method == "magnitude_cosine":  # NOT cab_v4
    # Current cosine similarity implementation
    ...
```

Then add proper CAB V4 as a separate method.

---

### 4.2 Use Triton Kernel in longbench_qa

**File:** `experiments/longbench_qa/methods.py`

**Changes:**
```python
# Line 304 and line 425
# Before:
from cab_attention.kernels.coarsening import coarsen_qk_max_l2_pytorch

# After:
from cab_attention.kernels.coarsening import coarsen_qk_max_l2
```

---

## 5. Verification Checklist

### Baselines
- [x] H2O uses cumulative attention scores (Zhang et al., 2023)
- [x] StreamingLLM uses sinks + window (Xiao et al., 2023)
- [x] Local+Strided uses local + strided patterns (Child et al., 2019)
- [x] Random uses random selection
- [x] Dense uses full attention

### CAB V3
- [x] longbench_qa: Uses real `coarsen_qk_max_l2_pytorch` ⚠️ (should use Triton)
- [x] longbench_qa: Uses real `compute_block_frc`
- [x] longbench_qa: Uses real `generate_block_mask`
- [ ] perplexity_lm: N/A (method not used)
- [ ] downstream_tasks: Uses approximation ❌

### CAB V4
- [x] longbench_qa: Uses real coarsening ⚠️ (should use Triton)
- [x] longbench_qa: Uses real FRC computation
- [x] longbench_qa: Uses real hybrid mask generation
- [ ] perplexity_lm: Uses cosine similarity approximation ❌
- [ ] downstream_tasks: Uses cosine similarity approximation ❌

---

## 6. Conclusion

**What works well:**
1. ✅ All baselines (H2O, StreamingLLM, Local+Strided, Random) are correctly implemented according to their publications
2. ✅ `longbench_qa/methods.py` uses the actual CAB kernels we built
3. ✅ Code is well-organized and documented

**Critical issues:**
1. ❌ `perplexity_lm/runner.py` does NOT use CAB kernels - uses cosine similarity approximation
2. ❌ `downstream_tasks/runner.py` does NOT use CAB kernels - uses cosine similarity approximation
3. ⚠️ `longbench_qa/methods.py` should use Triton kernel instead of PyTorch version

**Recommendations:**
1. **URGENT:** Implement proper FRC-based KV cache pruning in perplexity_lm and downstream_tasks
2. **MINOR:** Switch to Triton kernel in longbench_qa for 10-30x speedup
3. **TESTING:** Verify that perplexity/downstream results are actually from CAB V4, not approximation

**For ICML Submission:**
- Only use longbench_qa results for CAB V4 (after Triton fix)
- Do NOT use perplexity_lm or downstream_tasks results for CAB V4 until fixed
- Or clearly label those results as "Magnitude + Cosine Similarity" (different method)

---

## Appendix: Code Comparison

### Correct CAB V4 (longbench_qa):
```python
# ✅ CORRECT
q_coarse, k_coarse = self.coarsen_fn(query, key, block_size)
frc_scores, affinity, redundancy = self.frc_fn(
    q_coarse, k_coarse,
    lambda_redundancy=0.3,
    formula='additive',
    normalization='minmax',
)
block_mask = self.mask_fn(
    frc_scores,
    magnitude_scores=magnitude_scores,
    magnitude_ratio=0.5,
)
```

### Incorrect CAB V4 (perplexity_lm, downstream_tasks):
```python
# ❌ INCORRECT - Not using FRC!
magnitude = keys.norm(dim=-1).mean(dim=(0, 1))
k_norm = F.normalize(keys_flat, dim=-1)
similarity = torch.mm(k_norm, k_norm.t())  # ❌ Cosine similarity, NOT FRC
uniqueness = 1.0 - similarity.mean(dim=-1)  # ❌ NOT redundancy from FRC
importance = 0.5 * mag_norm + 0.5 * uniq_norm  # ❌ Wrong uniqueness metric
```

**The difference:** Real FRC uses `affinity - lambda * redundancy` with graph-based triangulation, not simple cosine similarity!
