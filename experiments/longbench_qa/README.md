# LongBench QA Benchmark Suite

**ICML 2025 Benchmark for CAB-Attention**

Comprehensive long-context question answering benchmarks for evaluating sparse attention methods.

---

## Quick Start

```bash
# Quick test (10 samples, ~5 min)
python driver.py --quick-test

# Standard benchmark
python driver.py --preset standard

# Full ICML benchmark
python driver.py --preset full

# Custom configuration
python driver.py --datasets narrativeqa qasper --methods cab_v4 h2o --sparsity 0.9 0.95
```

---

## Datasets

### LongBench (THUDM/LongBench)
| Dataset | Task | Max Length | Metrics |
|---------|------|------------|---------|
| `narrativeqa` | QA | 16K | F1, ROUGE-L |
| `qasper` | Scientific QA | 8K | F1 |
| `multifieldqa_en` | Multi-domain QA | 8K | F1 |
| `hotpotqa` | Multi-hop QA | 8K | F1, EM |
| `2wikimqa` | Multi-doc QA | 8K | F1 |
| `musique` | Compositional QA | 8K | F1 |
| `gov_report` | Summarization | 16K | ROUGE |
| `qmsum` | Meeting Summarization | 16K | ROUGE |
| `multi_news` | Multi-doc Summarization | 8K | ROUGE |

### SCROLLS (tau/scrolls)
| Dataset | Task | Max Length | Metrics |
|---------|------|------------|---------|
| `quality` | Multiple Choice | 8K | Accuracy |
| `qasper_scrolls` | Scientific QA | 8K | F1 |
| `narrativeqa_scrolls` | Story QA | 16K | F1, ROUGE-L |
| `summ_screen_fd` | Dialogue Summarization | 8K | ROUGE |

### InfiniteBench (128K+ Context)
| Dataset | Task | Max Length | Metrics |
|---------|------|------------|---------|
| `passkey` | Passkey Retrieval | 128K | Accuracy |
| `number_string` | Number Retrieval | 128K | Accuracy |
| `kv_retrieval` | Key-Value Retrieval | 128K | Accuracy |
| `longbook_qa_eng` | Book QA | 128K | F1 |
| `longbook_sum_eng` | Book Summarization | 128K | ROUGE-L |

### ZeroSCROLLS (Zero-shot)
Same tasks as SCROLLS but evaluated in zero-shot setting.

---

## Methods

| Method | Type | Description |
|--------|------|-------------|
| `dense` | Oracle | Full attention (upper bound) |
| `h2o` | Magnitude | Heavy-Hitter Oracle - keeps high-attention tokens |
| `cab_v4` | **Hybrid** | **CAB V4 - magnitude + FRC (RECOMMENDED)** |
| `cab_v3` | Topology | Pure FRC selection |
| `streaming_llm` | Pattern | Attention sinks + recent window |
| `local_strided` | Pattern | Local window + strided attention |
| `random` | Baseline | Random selection (lower bound) |

---

## Usage Examples

### 1. Quick Test
```bash
python driver.py --quick-test
```

### 2. Sparsity Sweep
```bash
python driver.py --sweep \
    --datasets narrativeqa qasper \
    --methods cab_v4 h2o \
    --output-dir results/sparsity_sweep
```

### 3. Method Comparison
```bash
python driver.py --compare-all \
    --datasets narrativeqa \
    --sparsity 0.9
```

### 4. Using Config File
```bash
python driver.py --config configs/icml_full.yaml
```

### 5. Custom CAB Parameters
```bash
python driver.py \
    --datasets narrativeqa \
    --methods cab_v4 \
    --magnitude-ratio 0.5 \
    --lambda-redundancy 0.3 \
    --block-size 64 \
    --sparsity 0.9 0.95
```

### 6. Different Model
```bash
python driver.py \
    --model mistralai/Mistral-7B-v0.1 \
    --datasets narrativeqa \
    --methods cab_v4
```

---

## Configuration Files

Pre-configured experiments in `configs/`:

| Config | Description | Est. Time |
|--------|-------------|-----------|
| `quick_test.yaml` | Quick debugging test | ~5 min |
| `icml_full.yaml` | Complete ICML benchmark | ~10-20 hours |
| `ablation_sparsity.yaml` | Sparsity level ablation | ~2-4 hours |
| `ablation_magnitude_ratio.yaml` | CAB V4 ratio ablation | ~1-2 hours |
| `infinitebench.yaml` | 128K context tests | ~4-8 hours |

---

## Output Structure

```
results/<experiment_name>/
├── <experiment>_results.json     # Full results with per-sample data
├── <experiment>_summary.txt      # Human-readable summary
├── intermediate_*.json           # Per-method intermediate results
└── figures/                      # Generated plots (if available)
```

### Results JSON Schema
```json
{
  "name": "experiment_name",
  "method_results": {
    "dataset_method_sparsity": {
      "method_name": "cab_v4",
      "dataset_name": "narrativeqa",
      "sparsity": 0.9,
      "metrics": {
        "f1": {"mean": 0.75, "std": 0.12, "min": 0.3, "max": 0.95}
      },
      "sample_results": [...]
    }
  }
}
```

---

## Presets

Use `--preset <name>` for quick configuration:

| Preset | Datasets | Methods | Sparsity | Samples |
|--------|----------|---------|----------|---------|
| `quick` | 1 | 2 | 1 | 10 |
| `standard` | 3 | 4 | 2 | 100 |
| `full` | 8 | 7 | 6 | all |
| `longbench_qa` | 6 | 4 | 2 | 200 |
| `longbench_sum` | 3 | 3 | 2 | 100 |
| `scrolls` | 4 | 4 | 2 | 100 |
| `infinitebench` | 3 | 4 | 2 | 50 |
| `zeroscrolls` | 3 | 3 | 1 | 100 |
| `ablation_sparsity` | 2 | 1 | 11 | 100 |

---

## API Usage

```python
from experiments.longbench_qa import (
    BenchmarkRunner,
    ExperimentConfig,
    get_dataset,
    get_method,
    compute_metrics,
)

# Quick evaluation
from experiments.longbench_qa.runner import quick_evaluate

result = quick_evaluate(
    dataset_name="narrativeqa",
    method_name="cab_v4",
    sparsity=0.9,
    max_samples=10,
)
print(f"F1: {result.metrics['f1']['mean']:.4f}")

# Method comparison
from experiments.longbench_qa.runner import compare_methods_on_dataset

results = compare_methods_on_dataset(
    dataset_name="narrativeqa",
    methods=["dense", "h2o", "cab_v4"],
    sparsity=0.9,
)

for method, result in results.items():
    print(f"{method}: F1={result.metrics['f1']['mean']:.4f}")
```

---

## Command Reference

```
usage: driver.py [-h] [--name NAME] [--description DESCRIPTION]
                 [--preset {quick,standard,full,...}]
                 [--config CONFIG] [--quick-test] [--sweep] [--compare-all]
                 [--datasets DATASETS [DATASETS ...]]
                 [--methods METHODS [METHODS ...]]
                 [--sparsity SPARSITY [SPARSITY ...]]
                 [--magnitude-ratio MAGNITUDE_RATIO]
                 [--lambda-redundancy LAMBDA_REDUNDANCY]
                 [--block-size BLOCK_SIZE]
                 [--model MODEL] [--max-length MAX_LENGTH]
                 [--max-new-tokens MAX_NEW_TOKENS]
                 [--dtype {float16,bfloat16,float32}]
                 [--max-samples MAX_SAMPLES] [--batch-size BATCH_SIZE]
                 [--seed SEED] [--output-dir OUTPUT_DIR]
                 [--save-predictions] [--save-attention]
                 [-v] [-q] [--wandb] [--wandb-project WANDB_PROJECT]
                 [--dry-run] [--list-datasets] [--list-methods]

options:
  -h, --help            show this help message and exit
  --name, -n            Experiment name
  --preset, -p          Use preset configuration
  --config, -c          Path to YAML/JSON config file
  --quick-test          Run quick test
  --sweep               Run sparsity sweep
  --compare-all         Compare all methods
  --datasets            Datasets to evaluate
  --methods             Methods to evaluate
  --sparsity            Sparsity levels
  --magnitude-ratio     CAB V4 magnitude ratio (0=FRC, 1=magnitude)
  --lambda-redundancy   FRC lambda parameter
  --block-size          Block size for block-sparse attention
  --model               HuggingFace model name
  --max-length          Maximum context length
  --max-samples         Maximum samples per dataset
  --output-dir, -o      Output directory
  --dry-run             Print config and exit
  --list-datasets       List available datasets
  --list-methods        List available methods
```

---

## Requirements

```bash
pip install torch transformers datasets tqdm numpy
pip install rouge-score  # For ROUGE metrics
pip install pyyaml       # For YAML configs
```

---

## Citation

If you use this benchmark, please cite:

```bibtex
@article{cab-attention-2025,
  title={Geometric Inductive Biases for Hardware-Aware Sparse Attention},
  author={[Author]},
  journal={ICML},
  year={2025}
}
```

---

## License

MIT License

