"""
Language Model Perplexity Benchmark (TODO 1.3)

Provides ICML publication-quality benchmarking for:
- WikiText-103 (standard LM benchmark)
- C4 (diverse web text)
- PG-19 (long book sequences)

Supports:
- Perplexity vs context length scaling analysis
- Perplexity vs sparsity trade-off curves
- Multiple sparse attention methods comparison

Usage:
    # Quick test
    python -m experiments.perplexity_lm.driver --quick-test
    
    # Full benchmark
    python -m experiments.perplexity_lm.driver --icml-full

    # Programmatic
    from experiments.perplexity_lm import run_benchmark, create_quick_test
    config = create_quick_test()
    results = run_benchmark(config)
"""

from .config import (
    # Enums
    PerplexityDataset,
    MethodName,
    
    # Configurations
    DatasetConfig,
    MethodConfig,
    ModelConfig,
    ExperimentConfig,
    BenchmarkConfig,
    ContextLengthSweepConfig,
    SparsitySweepConfig,
    
    # Config registries
    DATASET_CONFIGS,
    METHOD_CONFIGS,
    
    # Factory functions
    create_quick_test,
    create_icml_benchmark,
    get_dataset_config,
    get_method_config,
)

from .data_loaders import (
    TextChunk,
    PerplexityDataset as PerplexityDatasetLoader,
    WikiTextDataset,
    C4Dataset,
    PG19Dataset,
    create_perplexity_dataset,
    create_dataloader,
    get_dataset_stats,
    load_all_datasets,
)

from .metrics import (
    PerplexityResult,
    PerplexityEvaluator,
    compute_perplexity,
    compute_sliding_window_perplexity,
    aggregate_results,
    format_perplexity_result,
    compute_perplexity_statistics,
)

from .runner import (
    PerplexityBenchmarkRunner,
    run_benchmark,
    analyze_context_scaling,
    analyze_sparsity_tradeoff,
    generate_summary_table,
)

__all__ = [
    # Enums
    'PerplexityDataset',
    'MethodName',
    
    # Configurations
    'DatasetConfig',
    'MethodConfig',
    'ModelConfig',
    'ExperimentConfig',
    'BenchmarkConfig',
    'ContextLengthSweepConfig',
    'SparsitySweepConfig',
    
    # Config registries
    'DATASET_CONFIGS',
    'METHOD_CONFIGS',
    
    # Factory functions
    'create_quick_test',
    'create_icml_benchmark',
    'get_dataset_config',
    'get_method_config',
    
    # Data loaders
    'TextChunk',
    'PerplexityDatasetLoader',
    'WikiTextDataset',
    'C4Dataset',
    'PG19Dataset',
    'create_perplexity_dataset',
    'create_dataloader',
    'get_dataset_stats',
    'load_all_datasets',
    
    # Metrics
    'PerplexityResult',
    'PerplexityEvaluator',
    'compute_perplexity',
    'compute_sliding_window_perplexity',
    'aggregate_results',
    'format_perplexity_result',
    'compute_perplexity_statistics',
    
    # Runner
    'PerplexityBenchmarkRunner',
    'run_benchmark',
    'analyze_context_scaling',
    'analyze_sparsity_tradeoff',
    'generate_summary_table',
]

