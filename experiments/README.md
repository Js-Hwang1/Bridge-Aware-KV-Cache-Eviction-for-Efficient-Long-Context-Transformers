# CAB Attention - ICML Experiments

This directory contains all experiments for the ICML 2025 submission.

## Experiments

### Experiment 1: NIAH Benchmark (`exp1_niah_cab_v4.py`)

**Purpose:** Main experiment comparing CAB V4 vs H2O on needle-in-a-haystack tasks.

**Tests:**
- H2O (magnitude-based baseline)
- CAB V4 variants with different magnitude ratios:
  - 0.0 (pure FRC)
  - 0.25 (25% magnitude, 75% FRC)
  - 0.5 (50/50 hybrid) - **RECOMMENDED**
  - 0.75 (75% magnitude, 25% FRC)
  - 1.0 (pure magnitude = H2O)

**Parameters:**
- Context lengths: 1K, 2K, 4K tokens
- Sparsity levels: 85%, 90%, 95%
- Needle depths: 0%, 25%, 50%, 75%, 100%
- Trials per config: 5

**Output:** `results/exp1_niah_results.json`

**Run:**
```bash
cd experiments
python exp1_niah_cab_v4.py
```

**Expected runtime:** ~30-60 minutes on GPU

---

### Experiment 2: Analyze Results (`exp2_analyze_results.py`)

**Purpose:** Analyze exp1 results and generate paper-ready tables.

**Outputs:**
- Statistical analysis by sparsity level
- LaTeX tables for paper
- Performance recommendations

**Run:**
```bash
cd experiments
python exp2_analyze_results.py
```

**Requirements:** exp1_niah_results.json must exist

---

### Validation Tests (`test_cab_icml.py`)

**Purpose:** Synthetic validation tests (discriminative power, formula comparison, etc.)

**Note:** These are for validation only, not paper data. Results documented in `../ICML_VALIDATION_RESULTS.md`

**Run:**
```bash
cd experiments
python test_cab_icml.py
```

---

## Results Directory

All experiment outputs are saved to `results/`:
- `exp1_niah_results.json` - NIAH benchmark data
- Future experiments will add more result files here

---

## Dataset Directories

- `longbench_qa/` - LongBench dataset for QA tasks
- `qasper_qa/` - Qasper dataset for QA tasks
- `quality_qa/` - QuALITY dataset for QA tasks

**Note:** These are placeholder directories. Actual datasets need to be downloaded separately.

---

## Quick Start - Run First Experiment

```bash
# Ensure you're in FRC root directory
cd /Users/j/Desktop/FRC

# Run NIAH experiment
python experiments/exp1_niah_cab_v4.py

# Analyze results
python experiments/exp2_analyze_results.py
```

---

## Paper Data Collection Status

- [x] Exp1: NIAH Benchmark - **READY TO RUN**
- [ ] Exp2: Topology-focused tasks (bridge detection, etc.)
- [ ] Exp3: Large-scale benchmarks (8K, 16K contexts)
- [ ] Exp4: Real LLM integration (Llama, GPT-style)

---

## Configuration

All experiments use the validated CAB V4 configuration from `../ICML_FINAL_RESULTS.md`:

```python
# FRC computation
formula = 'additive'
normalization = 'minmax'
lambda_redundancy = 0.3

# Hybrid selection
magnitude_ratio = 0.5  # 50/50 hybrid (RECOMMENDED)
block_size = 32
```

---

## Notes

- All experiments use `torch.manual_seed(42)` for reproducibility
- Results are JSON format for easy parsing and plotting
- LaTeX table generation included in analysis scripts
- GPU strongly recommended (CUDA required for Triton kernels)

---

**Last Updated:** November 26, 2024
**Status:** Ready for ICML paper data collection
