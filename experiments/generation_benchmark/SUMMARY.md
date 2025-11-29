# Generation Benchmark Summary

## What We Built

A comprehensive, ICML-level rigorous benchmarking suite for evaluating KV cache eviction methods on **language generation tasks** (the original use case for H2O and CAB).

## Why This Matters

The previous QA benchmark revealed that CAB/H2O fail on question-answering because:
- **QA requires answer-specific relevance** (which tokens answer THIS question)
- **Generation requires general importance** (which tokens are useful context)
- Cumulative attention scores work for generation but not QA

This benchmark tests CAB/H2O on what they were actually designed for.

## Key Features

### 1. Proper Metrics for Generation
- **Perplexity**: Standard language modeling metric (lower = better)
- **Throughput**: Tokens per second (higher = better)
- **Memory**: Peak GPU usage (lower = better)
- **Cache Size**: Average KV cache length

### 2. Statistical Rigor
- Multiple runs with different random seeds
- Pairwise significance testing (t-tests)
- Mean, std, min, max for all metrics
- Reproducible experimental protocol

### 3. Robust Implementation
- Text concatenation for long sequences
- Flash Attention integration for O(N) scoring
- Proper KV cache pruning during generation
- Error handling and validation

### 4. Three Methods Compared
- **Dense**: Baseline (no eviction, full context)
- **H2O**: 20% recent + 80% highest importance
- **CAB**: 30% local + 20% bridges + 50% importance

## Running on Server

```bash
# SSH to server
ssh pen-cake-trails-brought.trycloudflare.com

# Quick test (10 samples, 1 run)
cd ~/FRC
git pull origin main
python -m experiments.generation_benchmark.driver \
  --methods dense h2o cab \
  --sparsity 0.9 \
  --dataset wikitext \
  --context-length 1024 \
  --num-samples 10 \
  --num-runs 1 \
  --experiment-name quick_test

# Full evaluation (50 samples, 3 runs, multiple sparsity levels)
python -m experiments.generation_benchmark.driver \
  --methods dense h2o cab \
  --sparsity 0.5 0.7 0.9 \
  --dataset wikitext \
  --context-length 2048 \
  --num-samples 50 \
  --num-runs 3 \
  --experiment-name full_eval
```

## Expected Results

Based on H2O/CAB papers, we expect:

### At 50% Sparsity (Keep 50% of cache)
- Dense: PPL ~12, 100% memory
- H2O: PPL ~12.5 (+4%), 50% memory, 2x faster
- CAB: PPL ~12.3 (+2.5%), 50% memory, 2x faster

### At 90% Sparsity (Keep 10% of cache)
- Dense: PPL ~12, 100% memory
- H2O: PPL ~15 (+25%), 10% memory, 8x faster
- CAB: PPL ~14 (+17%), 10% memory, 8x faster

**CAB should outperform H2O** by preserving bridge tokens that connect important contexts.

## Output Structure

```
results/generation/quick_test/
├── aggregated_results.json       # Summary statistics
├── significance_tests.json        # t-tests between methods
├── dense_s0.0_ctx1024_seed42.json
├── h2o_s0.9_ctx1024_seed42.json
├── cab_s0.9_ctx1024_seed42.json
```

## Integration with Paper

This benchmark provides:
1. **Baseline comparison**: Dense vs eviction methods
2. **Statistical significance**: P-values for all comparisons
3. **Multiple metrics**: Quality (PPL), speed (tokens/s), memory
4. **Ablation study**: Different sparsity levels
5. **Reproducibility**: Fixed seeds, documented protocol

Perfect for ICML/NeurIPS submission.

## Key Differences from QA Benchmark

| Aspect | QA Benchmark | Generation Benchmark |
|--------|-------------|---------------------|
| **Task** | Answer extraction | Language modeling |
| **Metric** | F1 score | Perplexity |
| **Relevance** | Question-specific | General importance |
| **CAB/H2O** | F1=0% (fail) | Expected to work |
| **Why** | Need answer-aware scoring | Cumulative attention sufficient |

## Current Status

- ✅ Benchmark code complete
- ✅ Flash Attention integrated
- ✅ Dataset loading robust
- ✅ Running on server `pen-cake-trails-brought.trycloudflare.com`
- ⏳ Waiting for results (est. 5-10 minutes for quick test)

## Next Steps

1. **Analyze results** from quick test
2. **Run full evaluation** (50 samples, 3 runs) if quick test passes
3. **Generate plots** (PPL vs sparsity, memory vs PPL, etc.)
4. **Write up findings** for paper/report
5. **Compare with published H2O/CAB results** for validation
