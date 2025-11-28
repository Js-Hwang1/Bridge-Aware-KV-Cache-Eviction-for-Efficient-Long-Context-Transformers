# CAB V5: Production Implementation Plan

**Date:** 2025-11-27
**Goal:** Production-ready KV cache eviction using 3-component importance
**Status:** Design document for complete rewrite of `cab_attention/`

---

## 1. Core Scientific Design

### 1.1 Three-Component Importance

**Intuition:**
- **Local context**: Nearby tokens for fluency and coherence
- **Global importance**: Frequently attended tokens (H2O-style heavy hitters)
- **Bridge tokens**: Connectors between important contexts (LOW FRC)

**Budget allocation:**
$$
\frac{\#\text{local} + \#\text{bridges} + \#\text{importance}}{\text{cache\_size}} = 1 - \text{sparsity}
$$

For 90% sparsity (keep 10%):
```python
total_budget = cache_size * 0.10  # 10% of tokens

# Proposed allocation (tunable hyperparameters)
local_budget = total_budget * 0.3      # 30% → local context
bridge_budget = total_budget * 0.2     # 20% → bridges (LOW FRC)
importance_budget = total_budget * 0.5 # 50% → heavy hitters (H2O)
```

**Rationale:**
- Local (30%): Recent window for fluency
- Bridges (20%): Connectors to maintain reasoning chains
- Importance (50%): Dominant H2O-style selection (proven effective)

---

### 1.2 Forman-Ricci Curvature for Bridge Detection

**Formula (from Forman 1999, Sreejith et al. 2016):**
$$
F(e_{ij}) = \underbrace{4 w_{ij}}_{\text{Edge weight}} - \underbrace{(S_i + S_j)}_{\text{Node strength}} + \underbrace{3 T_{ij}}_{\text{Triangles}}
$$

Where:
- $w_{ij}$ = attention weight from token $i$ to token $j$
- $S_i = \sum_k w_{ik}$ = strength of node $i$ (total outgoing attention)
- $S_j = \sum_k w_{kj}$ = strength of node $j$ (total incoming attention)
- $T_{ij}$ = number of triangles through edge $(i,j)$: $\sum_k \min(w_{ik}, w_{kj})$

**Interpretation:**
- **LOW (negative) FRC**: Bottleneck, bridge between communities
- **HIGH (positive) FRC**: Well-connected, embedded in dense cluster (redundant)

**Selection:**
```python
# Select tokens with LOWEST FRC (bridges)
bridge_indices = torch.topk(frc_scores, k=bridge_budget, largest=False).indices
```

---

## 2. KV Cache Update Strategy

### 2.1 Dynamic Eviction Policy

**Challenge:**
- During generation, cache grows token-by-token
- Need to prune when cache exceeds budget
- Must maintain all three components (local, bridges, importance)

**Design: Incremental Update with Lazy Eviction**

```python
class CABKVCache:
    def __init__(self, max_size, sparsity=0.9):
        self.max_size = max_size
        self.keep_ratio = 1 - sparsity

        # Component budgets
        self.local_ratio = 0.3
        self.bridge_ratio = 0.2
        self.importance_ratio = 0.5

        # State tracking
        self.cumulative_attention = None  # H2O-style tracking
        self.frc_scores = None
        self.last_eviction_at = 0

    def update(self, new_key, new_value, attention_weights=None):
        """Add new token to cache, evict if needed."""
        # Append new KV
        self.keys = torch.cat([self.keys, new_key], dim=2)
        self.values = torch.cat([self.values, new_value], dim=2)
        cache_len = self.keys.shape[2]

        # Update importance tracking
        if attention_weights is not None:
            self._update_importance_scores(attention_weights)

        # Evict if over budget
        if cache_len > self.max_size:
            self._evict()

    def _evict(self):
        """Evict tokens to fit budget using 3-component strategy."""
        cache_len = self.keys.shape[2]
        keep_size = int(cache_len * self.keep_ratio)

        # Component 1: Local context (most recent K tokens)
        local_size = int(keep_size * self.local_ratio)
        local_indices = torch.arange(cache_len - local_size, cache_len)

        # Component 2: Global importance (H2O-style, highest cumulative attention)
        importance_size = int(keep_size * self.importance_ratio)
        if self.cumulative_attention is not None:
            # Exclude local indices (already kept)
            candidate_mask = torch.ones(cache_len, dtype=torch.bool)
            candidate_mask[local_indices] = False

            candidate_scores = self.cumulative_attention.clone()
            candidate_scores[~candidate_mask] = -float('inf')

            importance_indices = torch.topk(
                candidate_scores, k=importance_size, largest=True
            ).indices
        else:
            # Fallback: keep most recent after local
            importance_indices = torch.arange(
                cache_len - local_size - importance_size,
                cache_len - local_size
            )

        # Component 3: Bridges (LOWEST FRC among remaining)
        bridge_size = keep_size - local_size - importance_size
        if bridge_size > 0:
            # Exclude already-kept indices
            kept_mask = torch.zeros(cache_len, dtype=torch.bool)
            kept_mask[local_indices] = True
            kept_mask[importance_indices] = True

            # Compute FRC for remaining candidates
            frc_scores = self._compute_frc_scores()

            candidate_frc = frc_scores.clone()
            candidate_frc[kept_mask] = float('inf')  # Exclude kept indices

            bridge_indices = torch.topk(
                candidate_frc, k=bridge_size, largest=False  # LOWEST FRC
            ).indices
        else:
            bridge_indices = torch.tensor([], dtype=torch.long)

        # Combine all kept indices
        keep_indices = torch.cat([local_indices, importance_indices, bridge_indices])
        keep_indices = keep_indices.unique().sort().values

        # Prune cache
        self.keys = self.keys[:, :, keep_indices, :]
        self.values = self.values[:, :, keep_indices, :]

        # Prune tracking state
        if self.cumulative_attention is not None:
            self.cumulative_attention = self.cumulative_attention[keep_indices]
```

---

### 2.2 When to Evict?

**Strategy: Amortized Eviction**

Instead of evicting at every step:
```python
# Evict when cache is K% over budget (e.g., 110%)
eviction_threshold = self.max_size * 1.1

if cache_len > eviction_threshold:
    self._evict_to_target(self.max_size * 0.9)  # Evict to 90% of budget
```

**Rationale:**
- Reduces eviction frequency
- Amortizes FRC computation cost
- Allows batch processing

---

### 2.3 FRC Computation Efficiency

**Challenge:**
- FRC requires:
  1. Node strengths: $O(N^2)$ (sum over all edges)
  2. Triangle counting: $O(N^3)$ (enumerate all triplets)
- Too expensive to compute every token

**Solution: Incremental Approximation**

```python
def _compute_frc_scores(self):
    """Compute FRC efficiently using approximations."""
    # Use attention weights from recent forward pass
    # (cached during generation)
    attn = self.recent_attention  # [B, H, N_q, N_kv]

    # Aggregate across heads and queries
    attn_agg = attn.mean(dim=(0, 1))  # [N_kv] (mean attention received)

    # Node strengths (approximate from cached attention)
    S = attn_agg  # Incoming strength

    # Edge weights (use cached pairwise attention)
    # For KV cache, we approximate edges as:
    # w_ij ≈ similarity(K_i, K_j)
    K_norm = F.normalize(self.keys.mean(dim=(0, 1)), dim=-1)  # [N, D]
    W = torch.mm(K_norm, K_norm.t())  # [N, N] similarity matrix

    # Triangle counting (approximate via matrix cube)
    # T_ij ≈ (W @ W)[i, j]  (number of 2-hop paths)
    W_squared = torch.mm(W, W)
    T = W_squared.diagonal()  # Triangles per node (approx)

    # Forman-Ricci Curvature (vectorized)
    # For each edge (i, j): F_ij = 4*w_ij - (S_i + S_j) + 3*T_ij
    # We approximate by computing per-node FRC:
    F_node = 4 * S - 2 * S + 3 * T  # Simplified node-level FRC
    # → F_node = 2*S + 3*T

    return F_node
```

**Further optimization:**
- Only compute FRC every K evictions (e.g., K=10)
- Use stale FRC scores between evictions
- Amortize cost: $O(N^2)$ every K steps → $O(N^2 / K)$ per step

---

## 3. Production Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────┐
│           CAB V5 KV Cache Manager               │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │     KV Cache (Dynamic Size)             │   │
│  │  - keys: [B, H, N, D]                   │   │
│  │  - values: [B, H, N, D]                 │   │
│  └─────────────────────────────────────────┘   │
│                    │                            │
│  ┌─────────────────┴──────────────────────┐    │
│  │  Importance Tracking (incremental)     │    │
│  │  - cumulative_attention [N]            │    │
│  │  - frc_scores [N] (lazy update)        │    │
│  │  - last_eviction_step                  │    │
│  └────────────────────────────────────────┘    │
│                    │                            │
│  ┌─────────────────┴──────────────────────┐    │
│  │  Eviction Policy (3-component)         │    │
│  │  - Local: Keep recent K tokens         │    │
│  │  - Importance: Keep high-attn tokens   │    │
│  │  - Bridges: Keep LOW FRC tokens        │    │
│  └────────────────────────────────────────┘    │
│                                                 │
└─────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   HuggingFace                    Standalone API
   Integration                    (for custom models)
```

---

### 3.2 File Structure

**New directory structure:**
```
cab_attention/
├── __init__.py                    # Public API
├── cache/
│   ├── __init__.py
│   ├── base.py                    # BaseKVCache interface
│   ├── cab_cache.py               # CABKVCache (3-component)
│   ├── h2o_cache.py               # H2OKVCache (baseline)
│   └── streaming_cache.py         # StreamingLLMCache (baseline)
├── scoring/
│   ├── __init__.py
│   ├── importance.py              # H2O-style cumulative attention
│   ├── frc.py                     # Forman-Ricci Curvature
│   └── hybrid.py                  # Combined scoring
├── eviction/
│   ├── __init__.py
│   ├── policies.py                # Eviction strategies
│   └── budget.py                  # Budget allocation
├── integration/
│   ├── __init__.py
│   ├── huggingface.py             # HF transformers integration
│   └── vllm.py                    # vLLM integration (future)
├── kernels/                       # DEPRECATED (legacy support only)
│   ├── coarsening.py              # Keep for backwards compat
│   └── frc_kernel.py              # Keep for backwards compat
└── utils/
    ├── __init__.py
    ├── metrics.py                 # Performance tracking
    └── visualization.py           # Debug visualizations
```

---

### 3.3 Core API Design

**Goal:** Drop-in replacement for HuggingFace `Cache` classes

```python
from cab_attention import CABCache

# Usage with HuggingFace models
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Create CAB cache
cache = CABCache(
    max_cache_size=4096,
    sparsity=0.9,
    local_ratio=0.3,
    bridge_ratio=0.2,
    importance_ratio=0.5,
)

# Generate with CAB cache
outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=512,
    past_key_values=cache,  # Pass CAB cache
    use_cache=True,
)
```

---

## 4. Implementation Plan

### Phase 1: Core Cache Implementation (Week 1)

**Files to create:**
1. `cab_attention/cache/base.py` - Abstract interface
2. `cab_attention/cache/cab_cache.py` - Main CAB cache
3. `cab_attention/scoring/importance.py` - H2O-style tracking
4. `cab_attention/scoring/frc.py` - FRC computation

**Deliverables:**
- [ ] `CABCache` class with 3-component eviction
- [ ] Unit tests for cache operations
- [ ] Benchmarks: cache update time < 5ms/token

---

### Phase 2: HuggingFace Integration (Week 1-2)

**Files to create:**
1. `cab_attention/integration/huggingface.py` - Monkey-patch attention layers

**Approach:**
```python
# Option A: Use HF's Cache API (preferred)
class CABCache(Cache):
    """Compatible with transformers.Cache interface."""

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Implement Cache.update() interface
        ...

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Required by Cache interface
        ...

# Option B: Monkey-patch model (if Cache API insufficient)
def replace_attention_with_cab(model, cache_config):
    """Replace model's attention with CAB-enabled version."""
    for layer in model.model.layers:
        original_forward = layer.self_attn.forward

        def cab_forward(hidden_states, attention_mask, **kwargs):
            # Intercept attention, apply CAB cache
            ...

        layer.self_attn.forward = cab_forward
```

**Deliverables:**
- [ ] Works with Llama, Qwen, Mistral architectures
- [ ] Passes HF's generation tests
- [ ] No degradation on dense (sparsity=0) mode

---

### Phase 3: Efficiency Optimizations (Week 2)

**Optimizations:**

1. **Lazy FRC computation**
   ```python
   # Only compute FRC every K evictions
   if self.steps_since_frc_update > 10:
       self.frc_scores = self._compute_frc()
       self.steps_since_frc_update = 0
   ```

2. **Batched eviction**
   ```python
   # Evict multiple tokens at once (not one-by-one)
   if cache_len > threshold:
       evict_to_size = int(self.max_size * 0.9)
       self._batch_evict(evict_to_size)
   ```

3. **CUDA kernel for FRC** (if needed)
   - Triton kernel for triangle counting
   - Fused kernel for FRC computation
   - Target: < 2ms for N=4096 tokens

**Deliverables:**
- [ ] FRC computation < 2ms for 4K cache
- [ ] Total overhead < 10% vs dense generation
- [ ] Memory usage < 2x dense attention

---

### Phase 4: Validation & Tuning (Week 3)

**Experiments:**

1. **Ablation: Component ratios**
   ```python
   # Test different budget allocations
   configs = [
       (0.3, 0.2, 0.5),  # Default (30% local, 20% bridge, 50% importance)
       (0.2, 0.3, 0.5),  # More bridges
       (0.4, 0.1, 0.5),  # More local
       (0.3, 0.0, 0.7),  # No bridges (H2O+Local baseline)
   ]

   # Measure on HotpotQA, NIAH, perplexity
   ```

2. **Ablation: Eviction frequency**
   ```python
   # Test amortization strategies
   eviction_intervals = [1, 5, 10, 20]  # Tokens between evictions
   ```

3. **Comparison: CAB vs baselines**
   ```python
   methods = ["dense", "h2o", "streaming_llm", "cab_v5"]
   tasks = ["hotpotqa", "niah", "narrativeqa", "perplexity"]
   sparsities = [0.85, 0.90, 0.95, 0.99]
   ```

**Success criteria:**
- CAB V5 >= H2O on all tasks (at least match)
- CAB V5 > H2O on at least 2/4 tasks (show advantage)
- Overhead < 15% latency increase

**Deliverables:**
- [ ] Optimal component ratios identified
- [ ] Performance >= H2O baseline
- [ ] Efficiency benchmarks documented

---

### Phase 5: Documentation & Release (Week 3-4)

**Documentation:**
1. API reference (Sphinx docs)
2. Usage examples (Jupyter notebooks)
3. Tuning guide (how to set hyperparameters)
4. Performance benchmarks (tables, plots)

**Release:**
1. PyPI package: `pip install cab-attention`
2. GitHub repo with examples
3. HuggingFace integration tutorial

---

## 5. Efficiency Analysis

### 5.1 Time Complexity

**Per-token generation step:**

| Component | Operation | Complexity | Amortized |
|-----------|-----------|------------|-----------|
| Attention | Standard attention | $O(N \cdot D \cdot H)$ | - |
| H2O tracking | Accumulate scores | $O(N)$ | $O(N)$ |
| FRC computation | Triangle counting | $O(N^2)$ | $O(N^2 / K)$ (every K steps) |
| Eviction | Sort + select | $O(N \log N)$ | $O(N \log N / K)$ |
| **Total overhead** | - | - | **$O(N + N^2/K + N\log N/K)$** |

**With K=10 (evict every 10 tokens):**
- FRC: $O(N^2) / 10$ → Negligible for N < 4096
- Eviction: $O(N \log N) / 10$ → < 1ms

**Target:** < 10% overhead vs dense generation

---

### 5.2 Memory Complexity

**Storage:**
```python
# KV cache
cache_size = max_cache_len * (1 - sparsity)  # 10% for sparsity=0.9
kv_memory = 2 * B * H * cache_size * D * sizeof(float16)

# Tracking state
importance_memory = cache_size * sizeof(float32)  # Cumulative attention
frc_memory = cache_size * sizeof(float32)        # FRC scores

# Total overhead
overhead = importance_memory + frc_memory
         ≈ 2 * cache_size * 4 bytes
         ≈ 8 * cache_size bytes
         ≈ 1% of KV cache memory (for typical H, D)
```

**Compared to dense:**
- Dense: Stores all N tokens
- CAB: Stores 10% of N tokens + tracking (negligible)
- **Memory savings: ~90%**

---

### 5.3 Realistic Speedup

**Assumptions:**
- Sparsity = 90% (keep 10%)
- Sequence length = 8192
- Overhead = 10%

**Analysis:**

| Metric | Dense | CAB V5 | Speedup |
|--------|-------|--------|---------|
| KV cache size | 8192 | 819 | 10x smaller |
| Attention FLOPs | $O(8192^2)$ | $O(819 \times 8192)$ | ~8x fewer |
| Memory bandwidth | 100% | ~12% (10% cache + 2% overhead) | ~8x reduction |
| Wall-clock time | 100% | ~20% (attention) + 10% (overhead) | **~3-4x faster** |

**Realistic speedup: 3-4x for 90% sparsity**

(Less than 10x because of overheads: generation, non-attention ops, etc.)

---

## 6. Risk Mitigation

### Risk 1: FRC Still Doesn't Help

**Indicators:**
- Bridge component doesn't improve over H2O+Local
- Optimal bridge_ratio → 0

**Mitigation:**
- Make bridge_ratio=0 a valid config (degrades to H2O+Local)
- Try alternative bridge metrics:
  - Betweenness centrality
  - PageRank
  - Effective resistance

---

### Risk 2: FRC Computation Too Expensive

**Indicators:**
- FRC takes > 5ms even with K=10 amortization
- Overhead > 20%

**Mitigation:**
- Further approximate FRC (skip triangle counting, use only node strengths)
- Use learned predictor instead of analytical FRC
- Increase amortization (K=20, 50, 100)

---

### Risk 3: HuggingFace Integration Issues

**Indicators:**
- Cache API doesn't support custom eviction
- Model architectures incompatible

**Mitigation:**
- Fallback to monkey-patching attention layers
- Implement for specific models (Llama, Qwen) first
- Contribute upstream to HF if needed

---

## 7. Validation Plan

### 7.1 Unit Tests

```python
# tests/test_cab_cache.py

def test_eviction_maintains_budget():
    """Test that cache never exceeds max_size."""
    cache = CABCache(max_size=100, sparsity=0.9)
    for i in range(200):
        cache.update(new_key, new_value)
        assert cache.size <= 100

def test_component_ratios():
    """Test that components respect budget ratios."""
    cache = CABCache(local_ratio=0.3, bridge_ratio=0.2, importance_ratio=0.5)
    cache.fill(200)  # Fill cache
    cache._evict()   # Trigger eviction

    # Check ratios (with some tolerance)
    assert abs(cache.local_count / cache.size - 0.3) < 0.1
    assert abs(cache.bridge_count / cache.size - 0.2) < 0.1

def test_frc_selects_low_curvature():
    """Test that bridges have lower FRC than rejected tokens."""
    cache = CABCache(...)
    bridge_frc = cache.frc_scores[cache.bridge_indices].mean()
    rejected_frc = cache.frc_scores[cache.rejected_indices].mean()
    assert bridge_frc < rejected_frc
```

---

### 7.2 Integration Tests

```python
# tests/test_huggingface_integration.py

def test_generate_with_cab_cache():
    """Test that CAB cache works with HF generate()."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    cache = CABCache(max_cache_size=512, sparsity=0.9)

    outputs = model.generate(
        input_ids=tokenizer("Hello", return_tensors="pt").input_ids,
        max_new_tokens=50,
        past_key_values=cache,
    )

    assert outputs.shape[1] == 51  # Input + 50 generated

def test_equivalence_to_dense_at_zero_sparsity():
    """Test that sparsity=0 matches dense attention."""
    cache_dense = CABCache(sparsity=0.0)

    # Generate with CAB (sparsity=0)
    outputs_cab = model.generate(..., past_key_values=cache_dense)

    # Generate with standard (no cache pruning)
    outputs_standard = model.generate(...)

    assert torch.allclose(outputs_cab, outputs_standard)
```

---

### 7.3 Benchmark Tests

```python
# benchmarks/benchmark_speed.py

def benchmark_generation_speed():
    """Measure tokens/sec with different cache strategies."""
    configs = {
        "dense": None,
        "h2o": H2OCache(sparsity=0.9),
        "streaming_llm": StreamingLLMCache(sparsity=0.9),
        "cab_v5": CABCache(sparsity=0.9),
    }

    for name, cache in configs.items():
        start = time.time()
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            past_key_values=cache,
        )
        elapsed = time.time() - start

        tokens_per_sec = 512 / elapsed
        print(f"{name}: {tokens_per_sec:.1f} tokens/sec")
```

---

## 8. Hyperparameter Tuning Guide

### 8.1 Component Ratios

**Default (recommended):**
```python
local_ratio = 0.3
bridge_ratio = 0.2
importance_ratio = 0.5
```

**Task-specific tuning:**

| Task Type | Local | Bridge | Importance | Rationale |
|-----------|-------|--------|------------|-----------|
| Question Answering | 0.2 | 0.3 | 0.5 | Need bridges for multi-hop |
| Summarization | 0.4 | 0.1 | 0.5 | Need local context for fluency |
| Code generation | 0.3 | 0.2 | 0.5 | Balanced (default) |
| Chat | 0.4 | 0.0 | 0.6 | Recent context + important history |

**Tuning procedure:**
1. Start with default (0.3, 0.2, 0.5)
2. Grid search over ratios (step=0.1)
3. Measure on validation set
4. Select best-performing ratios

---

### 8.2 Eviction Frequency

**Trade-off:**
- More frequent → More accurate, higher overhead
- Less frequent → Less accurate, lower overhead

**Recommended:**
```python
eviction_interval = 10  # Evict every 10 tokens
```

**Tuning:**
- If overhead > 15%: Increase interval (20, 50)
- If performance drops: Decrease interval (5, 1)

---

### 8.3 Sparsity Level

**Recommended range:** 0.85 - 0.95

| Sparsity | Keep % | Use Case |
|----------|--------|----------|
| 0.85 | 15% | High accuracy required |
| 0.90 | 10% | Balanced (default) |
| 0.95 | 5% | Maximum speedup |
| 0.99 | 1% | Extreme (likely too aggressive) |

---

## 9. Migration from Current CAB

### 9.1 Deprecation Plan

**Current code (to deprecate):**
- `cab_attention/coarse_predictor.py` - Block-level attention
- `cab_attention/kernels/frc_kernel.py` - Old FRC implementation
- `cab_attention/kernels/coarsening.py` - Block coarsening

**Migration:**
1. Mark old modules as deprecated (warnings)
2. Keep for backwards compatibility (6 months)
3. Remove in v2.0.0

**Backwards compatibility:**
```python
# cab_attention/__init__.py

# New API (recommended)
from .cache import CABCache

# Old API (deprecated)
from .coarse_predictor import CoarseCurvaturePredictor  # Deprecated

import warnings
warnings.warn(
    "CoarseCurvaturePredictor is deprecated. Use CABCache instead.",
    DeprecationWarning
)
```

---

### 9.2 Code Reuse

**What to keep:**
- Triton coarsening kernel (for future block-level experiments)
- FRC formula (adapt for new token-level computation)
- Testing infrastructure

**What to rewrite:**
- Cache management (new 3-component design)
- Eviction policy (dynamic, not static block masks)
- Integration (HuggingFace Cache API, not attention layer replacement)

---

## 10. Timeline & Milestones

### Week 1: Core Implementation
- [ ] Day 1-2: `CABCache` skeleton + importance tracking
- [ ] Day 3-4: FRC computation + eviction policy
- [ ] Day 5: Unit tests + local testing

### Week 2: Integration & Optimization
- [ ] Day 1-2: HuggingFace integration
- [ ] Day 3-4: Efficiency optimizations (lazy FRC, batching)
- [ ] Day 5: Integration tests + benchmarks

### Week 3: Validation
- [ ] Day 1-2: Run on HotpotQA, NIAH (validate >= H2O)
- [ ] Day 3-4: Component ratio tuning
- [ ] Day 5: Full benchmark suite (LongBench, perplexity)

### Week 4: Polish & Documentation
- [ ] Day 1-2: Documentation + examples
- [ ] Day 3-4: Performance tuning
- [ ] Day 5: Release prep (PyPI, GitHub)

**Total:** 4 weeks to production-ready CAB V5

---

## 11. Open Questions

### Q1: How to get attention weights during generation?

**Challenge:**
- Need attention weights to compute FRC
- HF's `generate()` doesn't return per-token attention by default

**Solutions:**

**Option A: Use `output_attentions=True`**
```python
outputs = model.generate(
    ...,
    output_attentions=True,  # Return attention weights
    return_dict_in_generate=True,
)
# Access via outputs.attentions
```
**Downside:** Memory overhead (stores all attention)

**Option B: Hook into attention layers**
```python
# Register hook to capture attention
def attention_hook(module, input, output):
    cache.register_attention(output[1])  # Attention weights

for layer in model.model.layers:
    layer.self_attn.register_forward_hook(attention_hook)
```
**Downside:** Model-specific, fragile

**Option C: Approximate from KV cache**
```python
# Compute attention from cached K and current Q
Q_current = ...  # From current forward pass
K_cached = cache.keys
attention_approx = F.softmax(Q_current @ K_cached.transpose(-2, -1), dim=-1)
```
**Downside:** Approximate, may drift from actual attention

**Recommendation:** Start with Option C (approximate), validate with Option A

---

### Q2: Should we compute FRC per-head or aggregated?

**Options:**

**Per-head FRC:**
```python
# Compute FRC separately for each attention head
frc_scores = []  # [H, N] (per-head)
for h in range(H):
    frc_h = compute_frc(attention[:, h, :, :])
    frc_scores.append(frc_h)

# Aggregate (mean, max, or learned combination)
frc_final = torch.stack(frc_scores).mean(dim=0)
```

**Aggregated FRC:**
```python
# Compute FRC on head-averaged attention
attention_agg = attention.mean(dim=1)  # [B, N, N]
frc_scores = compute_frc(attention_agg)  # [N]
```

**Recommendation:** Start with aggregated (simpler, faster), try per-head if needed

---

### Q3: How to handle variable-length sequences in batch?

**Challenge:**
- Batch of sequences with different lengths
- Each sequence has different cache state

**Solution: Per-sample caching**
```python
class CABCache:
    def __init__(self, ...):
        # Cache per batch element
        self.caches = [SingleCABCache() for _ in range(batch_size)]

    def update(self, key_states, value_states, layer_idx):
        # Update each cache independently
        for b in range(self.batch_size):
            self.caches[b].update(
                key_states[b:b+1],
                value_states[b:b+1],
            )
```

**Alternative: Padding-aware caching**
- Use attention mask to identify padding
- Don't evict padding tokens (or treat separately)

---

## 12. Success Metrics

### 12.1 Performance (Accuracy)

**Minimum bar (must achieve):**
- CAB V5 >= H2O on HotpotQA (F1 >= 0.0692)
- CAB V5 >= H2O on NIAH (Recall >= H2O baseline)
- CAB V5 >= H2O on perplexity (PPL <= H2O baseline)

**Target (to claim contribution):**
- CAB V5 > H2O on at least 2/4 benchmarks (show advantage)
- Statistical significance (p < 0.05)

---

### 12.2 Efficiency

**Target:**
- Latency overhead < 15% vs dense generation
- Memory savings ~90% (for 90% sparsity)
- Throughput: 3-4x tokens/sec improvement

**Measurement:**
- Benchmark on A100 GPU, batch_size=1, seq_len=8192
- Compare: Dense, H2O, StreamingLLM, CAB V5

---

### 12.3 Usability

**Target:**
- Drop-in replacement for HF Cache (< 5 lines of code change)
- Works with Llama, Qwen, Mistral, GPT architectures
- Clear documentation + examples

**Measurement:**
- User study: Can a new user integrate CAB in < 30 minutes?
- GitHub stars, PyPI downloads (post-release)

---

## 13. Conclusion

**This document provides:**
1. ✅ Rigorous 3-component design (local + bridges + importance)
2. ✅ Production-ready architecture (HF integration)
3. ✅ Efficiency analysis (3-4x speedup target)
4. ✅ Implementation plan (4 weeks to release)
5. ✅ Validation strategy (must match/beat H2O)

**Next steps:**
1. Review and approve this design
2. Start Week 1 implementation (CABCache core)
3. Validate on HotpotQA by end of Week 3
4. Only proceed to full benchmarks if CAB V5 >= H2O

**Key difference from CAB V4:**
- ❌ CAB V4: Block-level, select HIGH FRC, static
- ✅ CAB V5: Token-level, select LOW FRC (bridges), dynamic eviction

**Critical insight:** We need bridges (LOW FRC), not peripheral nodes (HIGH FRC)!
