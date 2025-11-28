# CAB V4: Scientific Issues & Proposed Improvements

**Date:** 2025-11-27
**Context:** CAB V4 (F1=0.0514) underperforms H2O (F1=0.0692) on HotpotQA @ 90% sparsity
**Goal:** Fix the fundamental scientific foundation before expensive GPU runs

---

## 1. Current Results Analysis

### HotpotQA Performance (Qwen2.5-7B, 8K context, 90% sparsity)
```
Dense:   F1 = 0.250  (upper bound)
H2O:     F1 = 0.0692 (magnitude baseline)
CAB V4:  F1 = 0.0514 (our method) ❌ WORSE than H2O
```

**What this tells us:**
- CAB V4's topological selection is HURTING performance, not helping
- The hybrid 50/50 split means we're sacrificing 50% of H2O's good magnitude selection
- The FRC component (the other 50%) is selecting the WRONG blocks
- We're not getting any benefit from the topological information

---

## 2. Fundamental Scientific Problems

### Problem 1: **FRC Measures Structure, Not Semantics**

**Issue:**
- Forman-Ricci Curvature measures **graph topology** (local connectivity patterns)
- Attention graphs represent **semantic similarity** (content-based relationships)
- These are NOT the same thing!

**Example failure mode:**
```
Token: "the" (common word, high redundancy, LOW FRC)
Token: "photosynthesis" (rare word, low redundancy, HIGH FRC)

FRC says: Keep "photosynthesis" (high curvature = unique)
Semantics say: Might not be relevant to the query at all!
```

**The mismatch:**
- Low FRC ≈ "structurally central" (bridge between communities)
- High FRC ≈ "structurally unique" (isolated or peripheral)
- Neither directly correlates with "semantically important for this query"

---

### Problem 2: **Block-Level vs Token-Level Granularity Mismatch**

**Issue:**
- FRC computed at BLOCK level (32x32 tokens)
- Semantic importance is TOKEN level
- Coarsening loses critical information

**Example:**
```
Block A: ["The", "quick", "brown", ..., "fox", "jumps", "over", ...]  (32 tokens)
  Contains: 1 important token "fox", 31 filler tokens

Block B: ["The", "the", "a", "an", ..., "is", "was", "are", ...]  (32 tokens)
  Contains: 0 important tokens, all filler

Max-L2 pooling on Block A and B might look similar!
Selecting/rejecting entire blocks is too coarse.
```

**The problem:**
- Important needles get averaged with haystack at block level
- Block-level FRC can't distinguish "block with 1 important token" from "block with 0 important tokens"

---

### Problem 3: **FRC Formula is Theoretically Weak**

**Current formula:**
```
FRC(i→j) = affinity(i,j) - λ * redundancy(i,j)
```

**Issues:**

1. **Additive combination is arbitrary**
   - Why subtract? Why not divide, multiply, or other operators?
   - No theoretical justification for linear combination
   - Lambda is a free parameter with no principled choice

2. **Redundancy term is oversimplified**
   ```python
   redundancy(i,j) = Σ_k affinity(i,k) * affinity(k,j)
   ```
   - This is just triangle counting weighted by edge weights
   - Assumes: More triangles = more redundancy
   - But in attention: More triangles might mean "strongly clustered important region"!

3. **No query dependence**
   - FRC is computed ONLY from Q_coarse and K_coarse
   - Doesn't consider what the CURRENT QUERY actually needs
   - Static topology vs dynamic query-dependent importance

---

### Problem 4: **Select HIGH FRC is Wrong Strategy**

**Current approach:**
```python
block_mask = generate_block_mask(
    frc_scores,
    select_high=True,  # ❌ Selecting HIGH curvature
)
```

**The issue:**
- HIGH FRC = structurally unique, isolated, peripheral
- LOW FRC = structurally central, bridge-like, well-connected
- For attention, we probably want CENTRAL nodes (low FRC), not peripheral ones!

**But wait:**
- Earlier validation showed selecting HIGH FRC worked on needle-in-haystack
- This suggests the relationship is task-dependent
- We don't have a consistent theory for when to select HIGH vs LOW

---

### Problem 5: **No Theoretical Guarantee**

**Missing:**
- No proof that FRC correlates with semantic importance
- No proof that hybrid magnitude+FRC beats pure magnitude
- No analysis of when FRC helps vs hurts
- No characterization of what types of tokens get selected

**What we have:**
- Empirical observation that it worked on one synthetic task
- Hand-wavy intuition about "bridges" and "unique connections"
- Mathematical inductiveness (it's a valid curvature measure) but no relevance proof

---

## 3. Why CAB V4 Fails on HotpotQA

**HotpotQA characteristics:**
- Multi-hop reasoning (need to connect facts across documents)
- Long contexts (8K tokens)
- Sparse relevant information
- Requires preserving multiple evidence chains

**Why FRC hurts here:**

1. **FRC selects "unique" blocks, not "important" blocks**
   - Unique ≠ Important for the query
   - Might select rare words that are irrelevant

2. **Block granularity loses fine-grained evidence**
   - Each hop might be 1-2 tokens in a 32-token block
   - Coarsening averages them away

3. **No query dependence**
   - FRC is computed globally, doesn't know what the question is asking
   - Might preserve topologically interesting but semantically irrelevant blocks

4. **Hybrid dilutes H2O's effectiveness**
   - H2O's magnitude selection is quite good (F1=0.0692)
   - Replacing 50% of it with poorly-chosen FRC blocks drops to F1=0.0514
   - We're paying a 26% relative drop in performance!

---

## 4. Proposed Scientific Improvements

### Improvement 1: **Query-Dependent Importance**

**Current (query-independent):**
```python
frc_scores = compute_block_frc(q_coarse, k_coarse)  # Static topology
```

**Proposed (query-dependent):**
```python
# Compute FRC conditioned on current query distribution
query_vector = q[:, :, -1, :]  # Current query token
frc_scores = compute_query_conditioned_frc(
    q_coarse, k_coarse,
    query_vector,  # What we're looking for
    attention_weights,  # Current attention pattern
)
```

**Idea:**
- FRC should measure "how important is this block FOR THIS QUERY"
- Not just "how topologically unique is this block in general"
- Weight affinity and redundancy by relevance to query

---

### Improvement 2: **Token-Level FRC (Not Block-Level)**

**Current:**
- Coarsen to blocks → Compute FRC → Select blocks → Expand to tokens
- Loses fine-grained information

**Proposed:**
```python
# Option A: Token-level FRC (computationally expensive)
frc_scores_token = compute_token_frc(q, k)  # [B, H, N, N]

# Option B: Block-level FRC + token-level refinement
block_frc = compute_block_frc(q_coarse, k_coarse)  # Coarse selection
token_importance = compute_token_importance_within_blocks(q, k, selected_blocks)  # Fine-grained

# Hybrid: Coarse block selection, then fine-grained token selection within selected blocks
```

**Idea:**
- Use blocks for computational efficiency (prune 90% of blocks)
- Use token-level analysis for final selection within kept blocks
- Two-stage: coarse-to-fine

---

### Improvement 3: **Better Redundancy Metric**

**Current redundancy (triangle counting):**
```python
redundancy(i,j) = Σ_k affinity(i,k) * affinity(k,j)
```

**Issues:**
- High triangles could mean "important clustered region" not "redundant region"

**Proposed alternatives:**

**Option A: Information-theoretic redundancy**
```python
# Mutual information between block i and block j
I(i, j) = H(i) + H(j) - H(i, j)
redundancy(i,j) = I(i, j)  # High MI = redundant
```

**Option B: Functional redundancy**
```python
# Can we remove block j without affecting attention from block i?
redundancy(i,j) = similarity(attn_with_j, attn_without_j)
```

**Option C: Content-based redundancy**
```python
# Semantic overlap (not just graph structure)
content_sim = cosine_similarity(embed_i, embed_j)
struct_redundancy = triangle_count(i, j)
redundancy = α * content_sim + β * struct_redundancy
```

---

### Improvement 4: **Learnable vs Fixed Formula**

**Current:**
```python
frc = affinity - λ * redundancy  # Fixed, hand-crafted
```

**Proposed:**

**Option A: Learn the combination**
```python
# Small MLP to combine affinity and redundancy
frc = MLP([affinity, redundancy, query_similarity, ...])
# Train end-to-end on downstream tasks
```

**Option B: Adaptive weighting**
```python
# Compute λ based on query properties
λ = adaptive_lambda(query_entropy, context_length, task_type)
frc = affinity - λ * redundancy
```

**Option C: Multi-scale FRC**
```python
# Compute FRC at multiple block sizes, combine
frc_8 = compute_frc(block_size=8)
frc_32 = compute_frc(block_size=32)
frc_128 = compute_frc(block_size=128)
frc_final = weighted_combine([frc_8, frc_32, frc_128])
```

---

### Improvement 5: **Theoretical Analysis First**

**Before running expensive experiments, answer:**

1. **What does FRC actually select?**
   - Run analysis on synthetic data where we KNOW the answer
   - Characterize: "Tokens with high FRC have property X"
   - Validate: "Property X correlates with importance for task Y"

2. **When does FRC help?**
   - Tasks where topology matters: Multi-hop, RAG, compositional reasoning
   - Tasks where topology doesn't matter: Simple QA, classification

3. **What's the optimal magnitude ratio?**
   - Theory: If magnitude has precision P_m and FRC has precision P_f
   - Optimal ratio = P_m / (P_m + P_f)
   - Estimate P_m and P_f empirically

---

## 5. Immediate Action Plan

### Step 1: **Diagnostic Analysis** (1-2 days)

**Goal:** Understand WHY CAB V4 fails on HotpotQA

**Experiments:**
1. **Visualization:**
   - Plot attention patterns: Dense vs H2O vs CAB V4
   - Highlight: Which tokens get selected/rejected by each method
   - Identify: Did CAB V4 select the wrong blocks?

2. **Error analysis:**
   - Sample 10-20 failed examples
   - Manual analysis: What information did CAB V4 miss?
   - Pattern: Is it always specific types of tokens? (entities, verbs, etc.)

3. **Component ablation:**
   ```python
   # Test each component separately
   cab_magnitude_only:  magnitude_ratio=1.0  (should match H2O)
   cab_frc_only:        magnitude_ratio=0.0  (pure topology)
   cab_hybrid_25:       magnitude_ratio=0.75 (mostly magnitude)
   cab_hybrid_50:       magnitude_ratio=0.5  (current)
   ```
   **Hypothesis:** If `cab_frc_only` is terrible, FRC is the problem

4. **FRC distribution analysis:**
   - Histogram of FRC scores
   - Correlation: FRC vs attention magnitude
   - Correlation: FRC vs token importance (human annotated)

---

### Step 2: **Fix Formula** (2-3 days)

**Based on diagnostics, try:**

**Option A: Swap selection strategy**
```python
# If selecting HIGH FRC is wrong, try LOW
block_mask = generate_block_mask(frc_scores, select_high=False)
```

**Option B: Query-conditioned FRC**
```python
# Weight FRC by query-key similarity
query_key_sim = torch.matmul(q_coarse[:, :, -1:, :], k_coarse.transpose(-2, -1))
frc_weighted = frc_scores * query_key_sim.squeeze(2)
```

**Option C: Two-stage selection**
```python
# Stage 1: Block-level (keep 20% blocks via FRC or magnitude)
keep_blocks = top_k(frc_scores, k=0.2)

# Stage 2: Token-level within kept blocks (select 10% tokens)
keep_tokens_in_blocks = top_k(attention_within_blocks, k=0.1)

# Total: 20% * 10% = 2% kept (98% sparse)
```

---

### Step 3: **Validate on Multiple Tasks** (1 week)

**Test improved CAB V4 on:**
1. HotpotQA (current failure case)
2. NIAH (known success case)
3. NarrativeQA (long document understanding)
4. Perplexity (general language modeling)

**Success criteria:**
- CAB V4 >= H2O on at least 3/4 tasks
- CAB V4 > H2O on at least 1/4 tasks (showing some advantage)
- Understand theoretically WHY it works when it works

---

### Step 4: **Theory Development** (ongoing)

**Write rigorous analysis:**

1. **Theorem 1:** Conditions under which FRC correlates with importance
   - Proof or counterexample

2. **Theorem 2:** Optimal magnitude ratio as function of task properties
   - Formal derivation

3. **Lemma:** Relationship between block-level FRC and token-level importance
   - Bounds on information loss from coarsening

---

## 6. Alternative Approaches to Consider

If FRC continues to underperform, consider:

### Alternative 1: **Learned Sparse Attention**

Instead of hand-crafted curvature, learn what to keep:
```python
class LearnedSelector(nn.Module):
    def forward(self, q, k, v):
        # Small predictor network
        importance = self.predictor(q, k)  # [B, H, N]
        mask = top_k(importance, k=sparsity_budget)
        return masked_attention(q, k, v, mask)

# Train end-to-end on downstream tasks
```

---

### Alternative 2: **Multi-Head Specialization**

Different heads use different strategies:
```python
# Head 0-3: Magnitude-based (like H2O)
# Head 4-5: FRC-based (topology)
# Head 6-7: Recency-based (like StreamingLLM)

# Aggregate outputs from all heads
```

---

### Alternative 3: **Hierarchical Attention**

```python
# Level 1: Coarse block selection (FRC-based)
blocks_to_keep = select_via_frc(q_coarse, k_coarse)

# Level 2: Fine token selection within blocks (magnitude-based)
tokens_to_keep = select_via_magnitude(q, k, within_blocks=blocks_to_keep)

# Two-stage: Coarse FRC + Fine Magnitude
```

---

## 7. Key Insights We Need to Gain

Before claiming scientific contribution, we must answer:

1. **What does FRC measure in attention contexts?**
   - Not just "it's a curvature measure"
   - Concrete interpretation for attention graphs

2. **When does FRC outperform magnitude?**
   - Characterize task types, data properties
   - Theoretical prediction, not just empirical trial

3. **How should we combine topology and magnitude?**
   - Optimal combination rule
   - Theoretical justification

4. **Can we prove any guarantees?**
   - Approximation bounds
   - Sample complexity
   - Convergence properties

---

## 8. Red Flags to Watch For

**Signs the approach is fundamentally flawed:**

1. CAB V4 consistently worse than H2O across all tasks
   → FRC is selecting wrong blocks

2. magnitude_ratio=1.0 (pure H2O) always best
   → FRC adds no value

3. FRC scores have no correlation with importance
   → Using wrong metric

4. Performance degrades with more FRC weight
   → Topology is hurting

**If we see these, we need to:**
- Abandon current FRC formula
- Try alternative topological measures (betweenness, PageRank, etc.)
- Or abandon topology altogether, focus on other innovations

---

## 9. Timeline Estimate

**Phase 1: Diagnosis** (2-3 days)
- Visualizations, error analysis, ablations
- Understand failure mode

**Phase 2: Fix** (3-5 days)
- Implement top 3 improvements
- Test on HotpotQA

**Phase 3: Validation** (1 week)
- Test on multiple benchmarks
- Ensure consistent gains

**Phase 4: Theory** (1-2 weeks)
- Formalize why it works
- Write theorems/proofs

**Total: 2-3 weeks to solid scientific foundation**

---

## 10. Conclusion

**Current state:**
- CAB V4 underperforms H2O (0.0514 vs 0.0692 on HotpotQA)
- FRC-based selection is hurting, not helping
- No solid theory for why it should work

**Root causes:**
1. FRC measures structure, not semantics
2. Block-level granularity too coarse
3. No query dependence
4. Wrong selection strategy (HIGH vs LOW)
5. Weak theoretical foundation

**Path forward:**
1. Run diagnostics to understand failure
2. Try query-conditioned FRC
3. Try two-stage coarse-to-fine
4. Develop rigorous theory
5. If still failing, consider alternatives

**Don't run expensive GPU experiments until we have:**
- CAB V4 >= H2O on validation tasks
- Clear theory for when/why it works
- Diagnostic tools to debug failures

---

**The user is RIGHT:** We need stronger science before scaling up. Let's fix the foundation first.
