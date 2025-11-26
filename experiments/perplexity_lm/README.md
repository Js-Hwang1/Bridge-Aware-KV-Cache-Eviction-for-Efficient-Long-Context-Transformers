# Language Model Perplexity Benchmark (TODO 1.3)

ICML publication-quality benchmark for evaluating sparse attention methods on language model perplexity.

## Overview

This benchmark evaluates CAB-Attention (and baselines) on:
- **WikiText-103**: Standard LM benchmark (clean Wikipedia text)
- **C4**: Diverse web text (real-world distribution)
- **PG-19**: Long book sequences (extreme long-context)

With analysis for:
- **Perplexity vs Context Length**: How does performance scale?
- **Perplexity vs Sparsity**: What's the accuracy-compute trade-off?

## Quick Start

```bash
# Quick test (TinyLlama on WikiText-2)
python -m experiments.perplexity_lm.driver --quick-test

# Full ICML benchmark
python -m experiments.perplexity_lm.driver --icml-full

# Custom configuration
python -m experiments.perplexity_lm.driver \
    --datasets wikitext-103 \
    --methods dense h2o cab_v4 \
    --sparsity-levels 0.0 0.9 0.95 \
    --context-lengths 512 1024 2048 4096
```

## Configuration Files

| Config | Description |
|--------|-------------|
| `configs/quick_test.yaml` | Fast debugging (TinyLlama, small data) |
| `configs/icml_full.yaml` | Full benchmark (Llama-2-7B, all datasets) |
| `configs/ablation_sparsity.yaml` | Fine-grained sparsity sweep |

## Datasets

| Dataset | Description | Context | Samples |
|---------|-------------|---------|---------|
| WikiText-103 | Clean Wikipedia | 4K | ~250K |
| WikiText-2 | Smaller Wikipedia | 2K | ~36K |
| C4 | Web crawl text | 4K | 1K (sampled) |
| PG-19 | Project Gutenberg books | 16K | 100 |

## Methods

| Method | Description | Key Parameter |
|--------|-------------|---------------|
| Dense | Full attention (oracle) | N/A |
| H2O | Magnitude-based selection | sparsity |
| CAB V4 | Hybrid (magnitude + FRC) | magnitude_ratio |
| CAB V3 | Pure FRC | lambda_redundancy |
| StreamingLLM | Sinks + window | window_size |
| Local+Strided | Fixed patterns | local_window, stride |
| Random | Random selection | sparsity |

## Metrics

- **Perplexity (PPL)**: exp(cross-entropy) - lower is better
- **Bits per Token (BPT)**: CE / ln(2) - lower is better
- **Cross-Entropy (CE)**: Raw loss in nats

## Expected Results

For WikiText-103 at sparsity=0.9:

| Method | PPL | Δ vs Dense |
|--------|-----|------------|
| Dense | ~10.0 | baseline |
| H2O | ~12.0 | +20% |
| CAB V4 | ~11.5 | +15% |
| CAB V3 | ~12.5 | +25% |

*CAB V4 should outperform H2O, showing topology helps!*

## Output

Results are saved to `results/perplexity/` as JSON:

```json
{
  "config": {...},
  "results": {
    "wikitext-103": {
      "cab_v4": {
        "context_length_sweep": {
          "512": {"perplexity": 15.2, ...},
          "1024": {"perplexity": 12.3, ...}
        },
        "sparsity_sweep": {
          "0.0": {"perplexity": 10.0, ...},
          "0.9": {"perplexity": 11.5, ...}
        }
      }
    }
  }
}
```

## Figures (for Paper)

This benchmark generates data for:
- **Figure 4**: Perplexity vs Context Length (scaling curves)
- **Figure 8**: Perplexity vs Sparsity (Pareto frontier)
- **Table 3**: Method comparison summary

## API Usage

```python
from experiments.perplexity_lm import (
    run_benchmark,
    create_quick_test,
    PerplexityBenchmarkRunner,
)

# Quick test
config = create_quick_test()
results = run_benchmark(config)

# Custom configuration
from experiments.perplexity_lm import ExperimentConfig, ModelConfig

config = ExperimentConfig(
    datasets=["wikitext-103"],
    methods=["dense", "h2o", "cab_v4"],
    model=ModelConfig(name="meta-llama/Llama-2-7b-hf"),
)
results = run_benchmark(config)
```

## Requirements

```bash
pip install torch transformers datasets tqdm pyyaml
```

## Citation

If you use this benchmark, please cite:

```bibtex
@inproceedings{cab2025,
  title={CAB: Curvature-Aware Block-Sparse Attention},
  author={...},
  booktitle={ICML},
  year={2025}
}
```

