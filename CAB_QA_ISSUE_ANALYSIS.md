# CAB Performance Issue on QA Tasks: Root Cause Analysis

## Summary

CAB (and H2O) achieve **F1=0.000** on HotpotQA at 90% sparsity, while Dense baseline gets **F1=60%**. After extensive debugging, the root cause is identified: **cumulative attention scores do not identify question-relevant tokens in QA tasks**.

## Timeline of Fixes

### 1. Flash Attention Bugs (FIXED ✓)
- **Bug 1**: Softmax padding mask - kernel produced wrong outputs for sequences < BLOCK_N
- **Bug 2**: NaN in cumulative scores - exp(-inf - (-inf)) = nan
- **Bug 3**: Score contamination - scores not reset between samples
- **Result**: Flash Attention now works correctly, produces coherent English

### 2. Eviction Policy Investigation

**Diagnostic Results** ([diagnose_eviction.py](diagnose_eviction.py)):

At 90% sparsity (keep 9/91 tokens) with `local_ratio=0.3`:
```
Top tokens (KEPT):
  Rank 1: "Context" (score=25674)
  Rank 2-20: "the", "Shiraz", ":", "?", "The", ".", etc.

Bottom tokens (EVICTED):
  "55", "km", "south", "Adelaide" (question specifics)
  "McLaren", "Vale" (answer tokens)

Answer tokens kept: 0/8
```

**Key Finding**: Both question AND answer are evicted!

### 3. Attempted Fix: Increase `local_ratio` to 0.7

**Rationale**: Keep more recent tokens to preserve the question (which appears at the end of the prompt before "Answer:").

**Configuration**:
```python
EvictionConfig(
    local_ratio=0.7,     # Keep 70% from recent (6-7 tokens)
    bridge_ratio=0.1,    # Keep 10% connectors (1 token)
    importance_ratio=0.2 # Keep 20% high attention (2 tokens)
)
```

**Result**: Still **F1=0.000**

**Sample Output**:
```
Prediction: "Mawson is an electoral district in the state of South Australia.
             It is located near the capital city of Adelaide, South Australia..."
Reference:  "McLaren Vale"
```

**Analysis**:
- ✅ Question is now preserved (model talks about "Mawson", "Adelaide", "electoral district")
- ❌ Answer ("McLaren Vale") is still evicted
- Problem: Answer tokens (positions 38-39) are neither recent (local) nor high importance

## Root Cause: Cumulative Attention ≠ Relevance for QA

### Why Cumulative Scores Fail on QA:

1. **Causal Attention During Prefill**:
   - When processing token at position 38 ("McLaren"), model can only attend to tokens 0-37
   - Model hasn't seen the question yet (positions 70-85)
   - Can't know "McLaren Vale" is the answer being asked for

2. **What Gets High Scores**:
   - Structural tokens: "Context:", "Question:", "Answer:" (attended to by many subsequent tokens)
   - First tokens (position bias in attention)
   - Generic prominent words: "the", "region", "wine"

3. **What Gets Low Scores**:
   - Specific facts: "55 km", "south", "McLaren Vale"
   - Question-specific information

4. **QA vs Generation**:
   - H2O/CAB papers evaluate on **generation** tasks (next-token prediction)
   - For generation, "important" = generally useful context
   - For QA, "important" = answers the specific question
   - These are different!

## Proposed Solutions

### Option 1: Question-Aware Scoring
After processing the full prompt (including question), boost importance scores for tokens semantically similar to question tokens.

```python
# Pseudo-code
question_embeddings = get_embeddings(question_tokens)
context_embeddings = get_embeddings(context_tokens)
similarity = cosine_similarity(context_embeddings, question_embeddings)
adjusted_scores = cumulative_attention + α * similarity
```

**Pros**: Directly addresses the problem
**Cons**: Requires computing embeddings (adds overhead)

### Option 2: Bi-directional Attention for Scoring
Compute importance using bidirectional attention (can see future tokens) for eviction decisions only.

**Pros**: Model can see question when scoring context
**Cons**: Breaks causality assumption, requires separate forward pass

### Option 3: Delay Eviction Until After First Generated Token
Don't evict during prefill. After generating first token, the model has attended to all prompt tokens - use those attention weights for eviction.

```python
# Pseudo-code
1. Forward pass on full prompt (no eviction)
2. Generate first token - capture attention weights from this step
3. Use attention weights to evict (model has "seen" what's important)
4. Continue generation with pruned cache
```

**Pros**: Model's attention reflects actual relevance after seeing question
**Cons**: First generation step uses full context (memory spike)

### Option 4: Dynamic Local Ratio Based on Prompt Structure
For QA tasks where question is at the end, automatically allocate more budget to local tokens.

```python
# If prompt ends with "Question: ... Answer:", use higher local_ratio
if is_qa_format(prompt):
    local_ratio = 0.9  # Keep almost everything recent
else:
    local_ratio = 0.3  # Default for generation
```

**Pros**: Simple, task-specific tuning
**Cons**: Doesn't solve fundamental issue - answer still may not be in recent tokens

### Option 5: Hybrid: Local + Semantic Similarity
Combine recent tokens (local) with semantically similar tokens to question.

```python
local_tokens = last_N_tokens(local_ratio * keep_size)
question_emb = mean(embeddings[question_tokens])
context_scores = cosine_similarity(embeddings[context_tokens], question_emb)
important_tokens = topk(context_scores, importance_ratio * keep_size)
keep_indices = union(local_tokens, important_tokens)
```

**Pros**: Balances recency and relevance
**Cons**: Still requires embedding computation

## Current Status

- ✅ Flash Attention kernel correctness
- ✅ Cumulative score numerical validity
- ✅ Score reset between samples
- ✅ Question preservation (with `local_ratio=0.7`)
- ❌ Answer preservation (fundamental scoring issue)
- ❌ F1 score improvement (still 0%)

## Recommendation

**Immediate**: Test **Option 3 (Delay Eviction)** - it's the least invasive change that could work with current infrastructure.

**Medium-term**: Implement **Option 5 (Hybrid Semantic)** for production QA workloads.

**Research**: Investigate whether original H2O/CAB papers tested on QA tasks or only generation tasks.

## Test Commands

```bash
# Current test (local_ratio=0.7)
sshpass -p '1234' ssh opposition-revolutionary-containers-empire.trycloudflare.com \
  "cd ~/FRC && git pull && python -m experiments.longbench_qa.driver \
   --model Qwen/Qwen2.5-7B-Instruct --methods cab --datasets hotpotqa \
   --sparsity 0.9 --max-samples 3"

# Diagnostic
sshpass -p '1234' ssh opposition-revolutionary-containers-empire.trycloudflare.com \
  "cd ~/FRC && git pull && python diagnose_eviction.py"
```

## References

- Flash Attention fixes: [FLASH_ATTENTION_FIX_SUMMARY.md](FLASH_ATTENTION_FIX_SUMMARY.md)
- Diagnostic script: [diagnose_eviction.py](diagnose_eviction.py)
- Position IDs fix: [POSITION_IDS_FIX_SUMMARY.md](POSITION_IDS_FIX_SUMMARY.md)
