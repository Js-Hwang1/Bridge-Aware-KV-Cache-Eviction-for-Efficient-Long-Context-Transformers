# ICML Experimental Plan: CAB-Attention Validation

**Goal**: Demonstrate that geometric (curvature-based) sparse attention outperforms magnitude-based methods on retrieval and reasoning tasks.

**Core Hypothesis**: Low-magnitude, high-importance "bridges" are critical for retrieval, and only CAB-Attention can preserve them.

---

## Tier 1: Core Validation Experiments (MUST HAVE for ICML)

### Experiment 1A: Needle-in-a-Haystack (Passkey Retrieval) ⭐⭐⭐
**The Killer Experiment** - This is your strongest evidence.

**Setup**:
- Context lengths: [4k, 8k, 16k, 32k, 64k, 128k]
- Needle positions: 10%, 25%, 50%, 75%, 90% depth
- Sparsity levels: [90%, 95%, 98%, 99%]
- Baselines: Full Attention, H2O, StreamingLLM, CAB (ours)

**Task**: Insert a random passkey (e.g., "The secret key is: 49281") into a long document of irrelevant text. Query: "What is the secret key?"

**Metric**: Exact match accuracy (0 or 1)

**Expected Result**:
- Full Attention: 100% (upper bound)
- H2O: Fails at 99% sparsity, especially for middle positions
- CAB: 100% even at 99% sparsity (preserves bridges)

**Visualization**: Heatmap (X=context length, Y=needle depth, Color=accuracy)

**Resources**: A100 40GB (sufficient for N=128k)

**Timeline**: 2-3 days

---

### Experiment 1B: Multi-Needle Retrieval ⭐⭐
**Extension** of NIAH to test compositional reasoning.

**Setup**:
- Insert K=2, 3, 5 needles at different positions
- Query requires retrieving ALL needles (e.g., "Sum all the numbers")
- Context: 32k, 64k, 128k

**Metric**: Accuracy (correct if all K needles retrieved)

**Expected Result**: CAB >> H2O (bridges connect disparate facts)

**Resources**: A100 40GB

**Timeline**: 1 day

---

### Experiment 1C: Reasoning Needle ⭐⭐⭐
**Most compelling** for reasoning tasks.

**Setup**:
- Insert chain-of-thought facts across long context
- Query requires multi-hop reasoning
- Example:
  - Fact 1 (position 10%): "Alice lives in Paris"
  - Fact 2 (position 50%): "Paris is in France"
  - Fact 3 (position 90%): "France uses the Euro"
  - Query: "What currency does Alice use?"

**Metric**: Exact match accuracy

**Expected Result**: CAB preserves reasoning chains; H2O drops critical links

**Resources**: A100 40GB

**Timeline**: 2-3 days

---

## Tier 2: Performance & Efficiency (IMPORTANT for Systems Track)

### Experiment 2A: Latency vs Accuracy Pareto Frontier ⭐⭐
**Show you're not just accurate but also fast.**

**Setup**:
- Vary sparsity from 50% to 99%
- Measure: Latency (ms) and NIAH accuracy
- Baselines: H2O, StreamingLLM, CAB
- Context: 64k, 128k

**Metric**: Plot accuracy (Y) vs latency (X)

**Expected Result**: CAB curve is "up and to the left" (better accuracy at same speed)

**Visualization**: Pareto frontier plot

**Resources**: A100 40GB

**Timeline**: 1 day

---

### Experiment 2B: Memory Efficiency ⭐
**Demonstrate practicality.**

**Setup**:
- Measure peak GPU memory for N=[4k, 8k, 16k, 32k, 64k, 128k]
- Compare: Full Attention, CAB-Attention
- Metric: Peak memory (GB), OOM threshold

**Expected Result**: CAB enables 128k on 40GB; Full Attention OOMs at 32k

**Resources**: A100 40GB

**Timeline**: 0.5 day

---

### Experiment 2C: Throughput (Tokens/sec) ⭐
**Wall-clock speedup.**

**Setup**:
- Batch size sweep: [1, 2, 4, 8, 16]
- Context: 16k, 32k, 64k
- Measure: Throughput (tokens/sec)

**Expected Result**: CAB achieves 2-5x throughput vs Full Attention at 64k+

**Resources**: A100 40GB

**Timeline**: 0.5 day

---

## Tier 3: Real-World Task Benchmarks (STRONG for Impact)

### Experiment 3A: Long-Context QA (LongBench) ⭐⭐⭐
**Standard benchmark for long-context models.**

**Dataset**: LongBench (NarrativeQA, Qasper, MultiFieldQA-en)

**Setup**:
- Use pretrained LLaMA-3-8B
- Swap attention mechanism: Full → CAB
- Sparsity: 95%
- Context: up to 64k (LongBench max)

**Metric**: F1 score, Exact Match

**Expected Result**: CAB ≈ Full Attention (no quality degradation)

**Resources**: A100 80GB (for LLaMA-3-8B + 64k context)

**Timeline**: 3-4 days (includes LLaMA integration)

---

### Experiment 3B: Document Summarization ⭐⭐
**Long-document understanding.**

**Dataset**: GovReport, SummScreen (from LongBench)

**Setup**:
- Context: Full documents (up to 32k tokens)
- Generate summaries with CAB vs baselines
- Metric: ROUGE-L

**Expected Result**: CAB ≈ Full Attention at 95% sparsity

**Resources**: A100 80GB

**Timeline**: 2 days

---

### Experiment 3C: Code Understanding (Long Code Files) ⭐⭐
**Novel application domain.**

**Dataset**: GitHub long files (Python, Java) with inserted bugs/queries

**Task**: "What does function X do?" (X is at random position in 50k-line file)

**Metric**: Accuracy of code understanding

**Expected Result**: CAB >> H2O (code dependencies are bridges)

**Resources**: A100 40GB

**Timeline**: 2-3 days

---

## Tier 4: Ablation Studies & Analysis (REQUIRED by Reviewers)

### Experiment 4A: FRC vs Magnitude Correlation ⭐⭐
**Prove they're different metrics.**

**Setup**:
- Compute both FRC and magnitude for all block pairs
- Scatter plot: X=magnitude, Y=FRC
- Highlight: Bridges (high FRC, low magnitude)

**Metric**: Pearson correlation (expect low r < 0.3)

**Expected Result**: Low correlation proves FRC captures orthogonal information

**Resources**: A100 40GB

**Timeline**: 0.5 day

---

### Experiment 4B: Lambda Hyperparameter Sweep ⭐
**Redundancy weighting.**

**Setup**:
- λ ∈ [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
- Run NIAH at 99% sparsity
- Metric: Retrieval accuracy

**Expected Result**: Optimal λ ≈ 0.5-1.0 (balances direct vs redundancy)

**Resources**: A100 40GB

**Timeline**: 1 day

---

### Experiment 4C: Block Size Ablation ⭐
**Coarsening granularity.**

**Setup**:
- Block sizes: [32, 64, 128, 256]
- Metric: NIAH accuracy, predictor latency

**Expected Result**: 64-128 optimal (balance between overhead and granularity)

**Resources**: A100 40GB

**Timeline**: 1 day

---

### Experiment 4D: Coarsening Strategy Comparison ⭐
**Max-L2 vs alternatives.**

**Setup**:
- Compare: Max-L2, Mean, Max-Magnitude, Random
- Metric: NIAH accuracy at 99% sparsity

**Expected Result**: Max-L2 > others (preserves needle signals)

**Resources**: A100 40GB

**Timeline**: 1 day

---

## Tier 5: Scaling Experiments (BONUS for H100/H200)

### Experiment 5A: Extreme Context (N=256k, 512k, 1M) ⭐⭐⭐
**Push the limits.**

**Setup**:
- Context lengths: [256k, 512k, 1M]
- NIAH at 99% sparsity
- Baselines: Only CAB can run (others OOM)

**Expected Result**: CAB works at 1M tokens on H200

**Resources**: H200 (141GB HBM3e)

**Timeline**: 2-3 days

---

### Experiment 5B: Multi-GPU Distributed Attention ⭐⭐
**Scalability.**

**Setup**:
- 2, 4, 8 GPUs
- Context: 512k, 1M tokens
- Measure: Latency, communication overhead

**Expected Result**: Near-linear scaling up to 4 GPUs

**Resources**: 4x H100

**Timeline**: 3-4 days

---

## Implementation Priority & Timeline

### Week 1: Core Validation (A100 40GB)
**Day 1-2**:
- ✅ Experiment 1A: Passkey Retrieval (THE critical experiment)
  - Implement dataset generation
  - Run CAB vs H2O comparison
  - Generate heatmap

**Day 3-4**:
- Experiment 1C: Reasoning Needle
- Experiment 4A: FRC vs Magnitude correlation

**Day 5**:
- Experiment 2A: Pareto frontier
- Experiment 2B: Memory efficiency

### Week 2: Real-World Tasks (A100 80GB)
**Day 6-8**:
- Experiment 3A: LongBench QA
  - Integrate with LLaMA-3-8B
  - Run on NarrativeQA, Qasper

**Day 9-10**:
- Experiment 3B: Document summarization
- Experiment 1B: Multi-needle

### Week 3: Ablations & Analysis
**Day 11-13**:
- All Experiment 4 ablations
- Generate analysis plots

### Week 4: Scaling (H100/H200)
**Day 14-16**:
- Experiment 5A: Extreme contexts (256k-1M)
- Final paper plots and tables

---

## Expected ICML Paper Structure

**Title**: "Geometric Inductive Biases for Hardware-Aware Sparse Attention"

**Abstract Claims** (backed by experiments):
1. "CAB achieves 100% retrieval at 99% sparsity where H2O fails" → Exp 1A
2. "Preserves reasoning chains in multi-hop QA" → Exp 1C
3. "No quality degradation on LongBench vs full attention" → Exp 3A
4. "Enables 1M-token contexts on H200" → Exp 5A
5. "< 5ms overhead for N=128k" → Already proven

**Figures** (6-8 total):
1. NIAH Heatmap (Exp 1A) - **THE money shot**
2. Pareto frontier (Exp 2A)
3. LongBench results table (Exp 3A)
4. FRC vs Magnitude scatter (Exp 4A)
5. Scaling curves (Exp 5A)
6. Qualitative example (bridge visualization)

---

## Resources Summary

| GPU Type | Experiments | Est. GPU-Days |
|----------|-------------|---------------|
| A100 40GB | 1A, 1B, 1C, 2A, 2B, 2C, 4A-4D | 10-12 days |
| A100 80GB | 3A, 3B, 3C | 7-8 days |
| H100/H200 | 5A, 5B | 5-6 days |
| **Total** | **All experiments** | **~20-25 GPU-days** |

**Cost Estimate** (cloud):
- A100 40GB: ~$1.50/hr × 12 days × 24hrs = ~$430
- A100 80GB: ~$2.50/hr × 8 days × 24hrs = ~$480
- H200: ~$5/hr × 6 days × 24hrs = ~$720
- **Total: ~$1,600** (very reasonable for ICML submission)

---

## Success Criteria (Acceptance Threshold)

**Minimum for ICML acceptance**:
- ✅ Exp 1A shows CAB >> H2O at 99% sparsity
- ✅ Exp 3A shows no quality loss vs full attention
- ✅ Exp 4A shows FRC ≠ Magnitude (uncorrelated)

**Strong acceptance** (add):
- ✅ Exp 1C shows reasoning preservation
- ✅ Exp 2A shows speed parity or improvement

**Outstanding acceptance / Oral** (add):
- ✅ Exp 5A demonstrates 1M-token capability
- ✅ Exp 3C shows novel application (code understanding)

---

## Starting Point: Experiment 1A Implementation

I'll now implement and run **Experiment 1A (Passkey Retrieval)** on your A100 via SSH.

This is the **highest priority** experiment - the one that will make or break the paper.

**Next**: Implement passkey retrieval benchmark and run CAB vs H2O comparison.
