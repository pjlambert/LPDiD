"""
Example: Using Parallel Processing in LP-DiD

This example demonstrates how to use the parallel processing capabilities
of the LPDiD package to speed up estimation for large datasets.
"""

import numpy as np
import pandas as pd
from LPDiD import LPDiD
import time

# Generate a large synthetic panel dataset
np.random.seed(123)

# Parameters
n_units = 2000
n_periods = 100
treatment_period = 50

# Create panel structure
data = []
for unit in range(n_units):
    for period in range(n_periods):
        # Treatment: half of units get treated at period 50
        treat = 1 if (unit < n_units // 2) and (period >= treatment_period) else 0
        
        # Generate outcome with treatment effect
        y = np.random.randn() + 0.01 * period
        if treat == 1:
            y += 1.5  # Treatment effect
        
        # Add some controls
        x1 = np.random.randn()
        x2 = np.random.randn()
        
        data.append({
            'unit': unit,
            'time': period,
            'treat': treat,
            'y': y,
            'x1': x1,
            'x2': x2
        })

df = pd.DataFrame(data)

print("Dataset created:")
print(f"- {len(df):,} observations")
print(f"- {n_units:,} units")
print(f"- {n_periods} time periods")
print(f"- Treatment starts at period {treatment_period}")

# Example 1: Sequential processing (default)
print("\n" + "="*60)
print("Example 1: Sequential Processing (n_jobs=1)")
print("="*60)

start_time = time.time()

lpdid_seq = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=10,
    post_window=20,
    formula='~ x1 + x2',
    ylags=5,  # Create 5 lags
    n_jobs=1  # Sequential processing
)

results_seq = lpdid_seq.fit()
seq_time = time.time() - start_time

print(f"\nSequential processing completed in {seq_time:.2f} seconds")

# Example 2: Parallel processing with all cores
print("\n" + "="*60)
print("Example 2: Parallel Processing (n_jobs=-1, all cores)")
print("="*60)

start_time = time.time()

lpdid_par = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=10,
    post_window=20,
    formula='~ x1 + x2',
    ylags=5,  # Create 5 lags
    n_jobs=-1  # Use all available cores
)

results_par = lpdid_par.fit()
par_time = time.time() - start_time

print(f"\nParallel processing completed in {par_time:.2f} seconds")
print(f"Speedup: {seq_time/par_time:.2f}x")

# Example 3: Parallel processing with specific number of cores
print("\n" + "="*60)
print("Example 3: Parallel Processing (n_jobs=4)")
print("="*60)

start_time = time.time()

lpdid_4cores = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=10,
    post_window=20,
    formula='~ x1 + x2',
    ylags=5,
    n_jobs=4  # Use 4 cores
)

results_4cores = lpdid_4cores.fit()
cores4_time = time.time() - start_time

print(f"\nProcessing with 4 cores completed in {cores4_time:.2f} seconds")
print(f"Speedup vs sequential: {seq_time/cores4_time:.2f}x")

# Compare results to ensure consistency
print("\n" + "="*60)
print("Verifying Results Consistency")
print("="*60)

# Check if coefficients are identical
seq_coef = results_seq.event_study['coefficient'].values
par_coef = results_par.event_study['coefficient'].values
cores4_coef = results_4cores.event_study['coefficient'].values

max_diff_par = np.max(np.abs(seq_coef - par_coef))
max_diff_4cores = np.max(np.abs(seq_coef - cores4_coef))

print(f"Max difference (sequential vs all cores): {max_diff_par:.2e}")
print(f"Max difference (sequential vs 4 cores): {max_diff_4cores:.2e}")

if max_diff_par < 1e-10 and max_diff_4cores < 1e-10:
    print("✓ All results are identical!")

# Display some results
print("\n" + "="*60)
print("Event Study Results (first 10 horizons)")
print("="*60)
print(results_par.event_study.head(10))

# Tips for using parallel processing
print("\n" + "="*60)
print("Tips for Using Parallel Processing")
print("="*60)
print("1. Use n_jobs=-1 to use all available CPU cores")
print("2. Parallel processing is most beneficial for:")
print("   - Large datasets (many units)")
print("   - Many horizons (large pre_window/post_window)")
print("   - Many lags (ylags > 2)")
print("3. For small datasets, sequential might be faster due to overhead")
print("4. Monitor CPU usage to ensure cores are being utilized")
print("5. Results are identical between sequential and parallel")
