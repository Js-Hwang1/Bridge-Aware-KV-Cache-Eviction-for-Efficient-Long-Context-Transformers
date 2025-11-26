#!/usr/bin/env python3
"""Quick analysis of EXP1 results without dependencies"""

import json
from collections import defaultdict
from statistics import mean, stdev

# Load results
with open('results/exp1_niah_results.json', 'r') as f:
    results = json.load(f)

print("=" * 80)
print("EXP1 NIAH Results Analysis")
print("=" * 80)
print()

# Aggregate by sparsity level
by_sparsity = defaultdict(lambda: {
    'dense': [],
    'h2o': [],
    'cab_0.00': [],
    'cab_0.25': [],
    'cab_0.50': [],
    'cab_0.75': [],
    'cab_1.00': []
})

# Aggregate compute times
compute_times = defaultdict(list)

for entry in results['data']:
    sparsity = entry['sparsity']

    for trial in entry['trials']:
        # Recall scores
        by_sparsity[sparsity]['dense'].append(trial['dense_recall'])
        by_sparsity[sparsity]['h2o'].append(trial['h2o']['recall'])
        by_sparsity[sparsity]['cab_0.00'].append(trial['cab_variants']['mag_0.00']['recall'])
        by_sparsity[sparsity]['cab_0.25'].append(trial['cab_variants']['mag_0.25']['recall'])
        by_sparsity[sparsity]['cab_0.50'].append(trial['cab_variants']['mag_0.50']['recall'])
        by_sparsity[sparsity]['cab_0.75'].append(trial['cab_variants']['mag_0.75']['recall'])
        by_sparsity[sparsity]['cab_1.00'].append(trial['cab_variants']['mag_1.00']['recall'])

        # Compute times
        compute_times['h2o'].append(trial['h2o']['time_ms'])
        compute_times['cab_0.00'].append(trial['cab_variants']['mag_0.00']['time_ms'])
        compute_times['cab_0.25'].append(trial['cab_variants']['mag_0.25']['time_ms'])
        compute_times['cab_0.50'].append(trial['cab_variants']['mag_0.50']['time_ms'])
        compute_times['cab_0.75'].append(trial['cab_variants']['mag_0.75']['time_ms'])
        compute_times['cab_1.00'].append(trial['cab_variants']['mag_1.00']['time_ms'])

# Print results by sparsity
print("RECALL BY SPARSITY LEVEL")
print("-" * 80)
print(f"{'Method':<15} {'85% Sparse':<15} {'90% Sparse':<15} {'95% Sparse':<15}")
print("-" * 80)

methods = [
    ('Dense', 'dense'),
    ('H2O', 'h2o'),
    ('CAB(0.00)', 'cab_0.00'),
    ('CAB(0.25)', 'cab_0.25'),
    ('CAB(0.50)', 'cab_0.50'),
    ('CAB(0.75)', 'cab_0.75'),
    ('CAB(1.00)', 'cab_1.00'),
]

for name, key in methods:
    s85 = mean(by_sparsity[0.85][key]) if by_sparsity[0.85][key] else 0
    s90 = mean(by_sparsity[0.90][key]) if by_sparsity[0.90][key] else 0
    s95 = mean(by_sparsity[0.95][key]) if by_sparsity[0.95][key] else 0
    print(f"{name:<15} {s85:>6.3f} ({len(by_sparsity[0.85][key]):<3} trials)  "
          f"{s90:>6.3f} ({len(by_sparsity[0.90][key]):<3} trials)  "
          f"{s95:>6.3f} ({len(by_sparsity[0.95][key]):<3} trials)")

print()
print("AVERAGE COMPUTE TIME (ms)")
print("-" * 80)
print(f"{'Method':<15} {'Mean':<12} {'Std Dev':<12}")
print("-" * 80)

compute_methods = [
    ('H2O', 'h2o'),
    ('CAB(0.00)', 'cab_0.00'),
    ('CAB(0.25)', 'cab_0.25'),
    ('CAB(0.50)', 'cab_0.50'),
    ('CAB(0.75)', 'cab_0.75'),
    ('CAB(1.00)', 'cab_1.00'),
]

for name, key in compute_methods:
    times = compute_times[key]
    if len(times) > 1:
        print(f"{name:<15} {mean(times):>8.2f} ms   {stdev(times):>8.2f} ms")
    elif len(times) == 1:
        print(f"{name:<15} {mean(times):>8.2f} ms   N/A")

print()
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Find best CAB variant
best_variant = None
best_recall = 0
for name, key in methods[2:]:  # Skip Dense and H2O
    avg_recall = mean([mean(by_sparsity[s][key]) for s in [0.85, 0.90, 0.95]])
    if avg_recall > best_recall:
        best_recall = avg_recall
        best_variant = name

h2o_avg = mean([mean(by_sparsity[s]['h2o']) for s in [0.85, 0.90, 0.95]])
h2o_time = mean(compute_times['h2o'])

print(f"1. Best CAB variant: {best_variant} with {best_recall:.3f} average recall")
print(f"2. H2O baseline: {h2o_avg:.3f} average recall")
print(f"3. H2O compute time: {h2o_time:.2f} ms")
print()

# Compare at each sparsity
print("COMPARISON AT EACH SPARSITY:")
for s in [0.85, 0.90, 0.95]:
    print(f"  {int(s*100)}% Sparse:")
    h2o_recall = mean(by_sparsity[s]['h2o'])
    print(f"    H2O: {h2o_recall:.3f}")
    for name, key in methods[2:]:
        cab_recall = mean(by_sparsity[s][key])
        diff = cab_recall - h2o_recall
        sign = "+" if diff >= 0 else ""
        print(f"    {name}: {cab_recall:.3f} ({sign}{diff:.3f})")
    print()

print("=" * 80)
