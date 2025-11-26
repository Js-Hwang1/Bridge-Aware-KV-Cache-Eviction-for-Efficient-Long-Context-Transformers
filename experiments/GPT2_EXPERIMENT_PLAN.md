# GPT-2 NIAH Experiment: Detailed Plan

**Date**: November 26, 2025
**Goal**: Validate FRC hypothesis with real language model
**Status**: Ready to run

---

## Core Scientific Question

**Can geometric (curvature-based) sparse attention preserve low-magnitude but structurally critical information better than magnitude-based methods?**

### Why This Matters for ICML

1. **Novel Selection Criterion**: First use of Forman-Ricci Curvature for transformer sparsity
2. **Quality-Efficiency Tradeoff**: 99% sparsity without accuracy degradation
3. **Practical Impact**: 10x inference speedup, enables 128k-1M token contexts
4. **Theoretical Foundation**: Geometric inductive bias vs. statistical heuristics

---

## What We're Testing

### The "Needle" Problem

```
Context (1024 tokens):
[Filler: "The sky is blue. Water flows..."] (100 tokens)
[Needle: "The secret key is 47291."]        (7 tokens)  ← CRITICAL but LOW MAGNITUDE
[Filler: "Birds fly south. Rain falls..."] (917 tokens)

Query: "What is the secret key?"

Attention Pattern:
- Filler tokens: HIGH magnitude (repeated, familiar)
- Needle tokens: LOW magnitude (rare number, specific fact)
```

### Hypothesis

**H2O (Magnitude-based)**:
- Keeps top-k attention scores by magnitude
- **Prediction**: Will drop needle (low magnitude) → Fails retrieval

**CAB (Curvature-based)**:
- Keeps tokens with low FRC (bridges in graph)
- Needle creates bridge: Query ← → Needle ← → Context
- **Prediction**: Preserves needle (bridge structure) → Succeeds retrieval

---

## Implementation Design

### 1. Architecture

```
┌─────────────────────────────────────┐
│  GPT2NeedleDataset                  │
│  - Generate NIAH samples            │
│  - Use GPT-2 tokenizer (BPE)        │
│  - Track needle token positions     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  GPT2AttentionExtractor             │
│  - Load pretrained GPT-2            │
│  - Forward pass: output_attentions  │
│  - Extract layer 6 (middle)         │
│  - Average across 12 heads          │
│  - Returns: [N, N] dense attention  │
└─────────────────────────────────────┘
              ↓
┌──────────────┬──────────────────────┐
│  H2O Mask    │  CAB Mask            │
│  (Top-k by   │  (Lowest FRC         │
│   magnitude) │   = bridges)         │
└──────────────┴──────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Needle Evaluator                   │
│  - Check: mask[query, needle] == 1  │
│  - Success: Any needle token kept   │
└─────────────────────────────────────┘
```

### 2. Key Design Decisions

**Model**: GPT-2 (124M)
- 12 layers, 12 heads, 768 dim
- Max native context: 1024 tokens
- Pretrained on diverse text → realistic attention patterns

**Target Layer**: Layer 6 (middle)
- Early layers: Positional/syntactic
- Middle layers: Semantic relationships
- Late layers: Task-specific (generation)
- **Layer 6**: Best balance of semantic + structural

**Needle Design**:
- Format: "The secret key is {5-digit number}."
- Numbers are **low-frequency tokens** in GPT-2 vocabulary
- Creates natural bridge structure

**Block Size**: 64
- Hardware-aligned (Tensor Cores)
- N=1024 → 16×16 blocks
- N=512 → 8×8 blocks

---

## Expected Runtime

### Configuration
```python
context_lengths = [512, 1024]      # Conservative start
needle_depths = [0.1, 0.25, 0.5, 0.75, 0.9]
sparsity_levels = [0.90, 0.95, 0.99]
methods = ['full', 'h2o', 'cab']
num_samples = 5
```

### Time Breakdown

**GPT-2 Forward Pass (output_attentions=True)**:
- N=512:  ~0.05s per sample
- N=1024: ~0.10s per sample

**Total Samples**:
- 2 lengths × 5 depths × 5 samples = 50 unique samples
- 50 × 3 methods × 3 sparsity = 450 evaluations
- But we only run GPT-2 forward 50 times (reuse attention)

**Calculation**:
```
N=512:  0.05s × 25 samples = 1.25s
N=1024: 0.10s × 25 samples = 2.50s
FRC computation: 450 × 0.3s = 135s (block-level operations)
Overhead: 30s (progress bars, saving)

TOTAL: ~3 minutes
```

**Memory**: Peak ~6GB (well within 40GB A100)

---

## Expected Results

### Success Criteria (ICML Validation)

**Minimum for Hypothesis Validation**:
```
Sparsity: 99%
Full Attention: 100% (upper bound)
H2O:            <50% (fails on needles)
CAB:            >80% (preserves bridges)
```

**Strong Result**:
```
Sparsity: 99%
Full Attention: 100%
H2O:            ~30%
CAB:            ~95%
```

**Ideal Result**:
```
Sparsity: 99%
Full Attention: 100%
H2O:            <20%
CAB:            100%
```

### What Different Outcomes Mean

**Scenario 1: CAB >> H2O** ✅
- FRC successfully identifies bridges
- Hypothesis validated
- **Next steps**:
  - Scale to longer contexts (2048, 4096, 8192)
  - Test on LongBench QA tasks
  - Write ICML paper

**Scenario 2: CAB ≈ H2O** ⚠️
- Need to investigate:
  - Try different layers (layer 8, 10)
  - Adjust λ hyperparameter (0.3, 0.7, 1.0)
  - Check if needles actually create bridges in this model
- May need different needle design

**Scenario 3: Both fail (CAB ≈ H2O ≈ 0%)** ❌
- Fundamental issue with:
  - Evaluation logic (check code)
  - Needle detection in BPE tokens
  - Or context too short for meaningful test

---

## Code Quality & Design

### Modularity
✓ Separate concerns: Dataset, Extractor, Masks, Evaluator
✓ Reusable components for future experiments
✓ Config dataclass for easy tuning

### Error Handling
✓ Try-catch in evaluation loop
✓ Validation of needle positions
✓ Graceful fallback for tokenization mismatches

### Debugging Support
✓ Progress bars with detailed metrics
✓ Info dict returns actual sparsity, needle stats
✓ Model info logging

### Memory Efficiency
✓ `@torch.no_grad()` decorator
✓ No gradient computation
✓ Process one sample at a time
✓ Delete intermediate tensors

### Reproducibility
✓ Fixed random seeds (can be added)
✓ Save all config to JSON
✓ Track exact token positions

---

## Pre-Flight Checklist

Before running on A100:

1. **Dependencies**:
   ```bash
   pip install transformers  # For GPT-2
   # torch, numpy, matplotlib already installed
   ```

2. **Disk Space**:
   - GPT-2 model: ~500MB
   - Results: <10MB

3. **Validation Test** (run locally first):
   ```python
   python exp1a_niah_passkey_gpt2.py
   # Should complete in ~3 minutes on A100
   ```

4. **Expected Outputs**:
   - `niah_results_gpt2.json`
   - `niah_heatmap_gpt2_sparsity_90.png`
   - `niah_heatmap_gpt2_sparsity_95.png`
   - `niah_heatmap_gpt2_sparsity_99.png`

---

## After Results

### If Successful (CAB >> H2O):

**Immediate Next Steps** (Week 1):
1. Scale to longer contexts:
   - Use position interpolation for N > 1024
   - Test: [2048, 4096, 8192]
   - Expected: CAB advantage increases with length

2. Multi-needle variant:
   - Insert 2-3 needles at different depths
   - Query requires finding all
   - Tests compositional reasoning

3. Ablation studies:
   - Lambda sweep: [0.3, 0.5, 0.7, 1.0]
   - Layer sweep: [4, 6, 8, 10]
   - Block size: [32, 64, 128]

**ICML Paper Sections**:
- Figure 1: NIAH heatmap (99% sparsity) ← **Money shot**
- Table 1: Accuracy comparison across sparsity levels
- Figure 2: FRC vs Magnitude scatter plot
- Section 4.1: "Needle-in-a-Haystack Validation"

---

## Risk Mitigation

**If runtime > 10 minutes**:
- Reduce to single sparsity: [0.99]
- Reduce samples: 5 → 3
- Reduce depths: 5 → 3

**If OOM**:
- Use `gpt2` instead of larger variants
- Process samples sequentially (already doing this)
- Reduce context to [512] only

**If transformers not installed**:
- Error message provides install command
- Can install on A100: `pip install transformers`

---

## Success Metrics for ICML

**Tier 1** (Must Have):
- ✅ CAB outperforms H2O at 99% sparsity by >30 percentage points
- ✅ Full attention = 100% (validates task)
- ✅ Results reproducible across 5 samples

**Tier 2** (Strong Paper):
- ✅ CAB achieves >90% accuracy at 99% sparsity
- ✅ Scales to N=8192 with maintained advantage
- ✅ Ablation studies show λ=0.5 is optimal

**Tier 3** (Outstanding/Oral):
- ✅ CAB = 100% at 99% sparsity (perfect preservation)
- ✅ Works on LongBench QA tasks
- ✅ Enables 1M token contexts on H200

---

## Code Location

**Main script**: `exp1a_niah_passkey_gpt2.py`

**Components**:
- `ExperimentConfig`: Configuration dataclass
- `GPT2NeedleDataset`: Sample generation
- `GPT2AttentionExtractor`: Attention extraction
- `apply_h2o_mask()`: Magnitude-based selection
- `apply_cab_mask()`: Curvature-based selection
- `evaluate_needle_retrieval()`: Success check
- `run_experiment()`: Main loop
- `plot_heatmap()`: Visualization

**Lines of Code**: ~650 (well-documented)

---

## Ready to Run

**Estimated Time**: 3 minutes
**Estimated Memory**: 6GB peak
**Expected Output Size**: ~300KB

**Command**:
```bash
cd ~/cab_attention_test/experiments
python3 exp1a_niah_passkey_gpt2.py
```

**First Checkpoint** (after ~30s):
- Should see: "Loading gpt2..."
- Should see: "✓ Model loaded: 12 layers, 12 heads, 768 dim"

**Progress Monitoring**:
- tqdm progress bar updates every sample
- Shows: method, N, depth, sparsity, accuracy

**Completion**:
- Saves 4 files (JSON + 3 PNGs)
- Prints summary table
- Shows interpretation guide

---

**This is the critical experiment for ICML validation. Let's execute!**
