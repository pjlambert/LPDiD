"""
Test script to demonstrate and benchmark parallel processing in LPDiD
"""

import numpy as np
import pandas as pd
import time
from LPDiD import LPDiD
import multiprocessing

# Generate synthetic panel data
def generate_test_data(n_units=1000, n_periods=50, seed=42):
    """Generate synthetic panel data for testing"""
    np.random.seed(seed)
    
    # Create panel structure
    units = []
    periods = []
    for i in range(n_units):
        units.extend([i] * n_periods)
        periods.extend(list(range(n_periods)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'unit': units,
        'time': periods
    })
    
    # Generate treatment (some units get treated at period 25)
    treated_units = np.random.choice(n_units, size=n_units//2, replace=False)
    df['treat'] = 0
    df.loc[(df['unit'].isin(treated_units)) & (df['time'] >= 25), 'treat'] = 1
    
    # Generate outcome with treatment effect
    df['y'] = np.random.randn(len(df)) + 0.1 * df['time']
    df.loc[df['treat'] == 1, 'y'] += 2.0  # Treatment effect
    
    # Add some control variables
    df['x1'] = np.random.randn(len(df))
    df['x2'] = np.random.randn(len(df))
    
    return df

def benchmark_lpdid(df, n_jobs_list=[1, 2, 4, -1], pre_window=5, post_window=10, ylags=5):
    """Benchmark LPDiD with different numbers of cores"""
    results = {}
    
    print("="*60)
    print("Benchmarking LP-DiD with Parallel Processing")
    print("="*60)
    print(f"\nDataset size: {len(df)} observations")
    print(f"Number of units: {df['unit'].nunique()}")
    print(f"Number of periods: {df['time'].nunique()}")
    print(f"Pre-window: {pre_window}, Post-window: {post_window}")
    print(f"Number of lags: {ylags}")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    print("\n" + "-"*60)
    
    for n_jobs in n_jobs_list:
        print(f"\nTesting with n_jobs = {n_jobs}")
        print("-"*30)
        
        start_time = time.time()
        
        # Run LP-DiD
        lpdid = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=pre_window,
            post_window=post_window,
            formula='~ x1 + x2 | unit',
            ylags=ylags,
            n_jobs=n_jobs
        )
        
        # Fit the model
        result = lpdid.fit()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        results[n_jobs] = {
            'time': elapsed_time,
            'result': result
        }
        
        print(f"Total time: {elapsed_time:.2f} seconds")
        
        # Show first few results
        print("\nFirst few event study estimates:")
        print(result.event_study.head())
    
    # Compare times
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    
    baseline_time = results[1]['time']
    for n_jobs, data in results.items():
        speedup = baseline_time / data['time']
        cores_used = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        print(f"n_jobs={n_jobs:2d} ({cores_used:2d} cores): {data['time']:6.2f}s (speedup: {speedup:.2f}x)")
    
    return results

def test_parallel_consistency(df, pre_window=5, post_window=10):
    """Test that parallel and sequential results are identical"""
    print("\n" + "="*60)
    print("Testing Parallel Processing Consistency")
    print("="*60)
    
    # Run sequential
    print("\nRunning sequential version...")
    lpdid_seq = LPDiD(
        data=df,
        depvar='y',
        unit='unit',
        time='time',
        treat='treat',
        pre_window=pre_window,
        post_window=post_window,
        formula='~ x1 + x2',
        ylags=3,
        n_jobs=1
    )
    result_seq = lpdid_seq.fit()
    
    # Run parallel
    print("\nRunning parallel version...")
    lpdid_par = LPDiD(
        data=df,
        depvar='y',
        unit='unit',
        time='time',
        treat='treat',
        pre_window=pre_window,
        post_window=post_window,
        formula='~ x1 + x2',
        ylags=3,
        n_jobs=-1
    )
    result_par = lpdid_par.fit()
    
    # Compare results
    print("\nComparing results...")
    
    # Compare event study coefficients
    seq_coef = result_seq.event_study['coefficient'].values
    par_coef = result_par.event_study['coefficient'].values
    
    max_diff = np.max(np.abs(seq_coef - par_coef))
    print(f"Maximum difference in coefficients: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✓ Results are identical!")
    else:
        print("✗ Results differ!")
        print("\nSequential coefficients:")
        print(seq_coef)
        print("\nParallel coefficients:")
        print(par_coef)
    
    return result_seq, result_par

if __name__ == "__main__":
    # Generate test data
    print("Generating test data...")
    df_small = generate_test_data(n_units=100, n_periods=30)
    df_large = generate_test_data(n_units=1000, n_periods=50)
    
    # Test consistency
    test_parallel_consistency(df_small)
    
    # Benchmark performance
    print("\n" + "="*60)
    print("Small Dataset Benchmark")
    benchmark_lpdid(df_small, n_jobs_list=[1, 2, 4, -1])
    
    print("\n" + "="*60)
    print("Large Dataset Benchmark")
    benchmark_lpdid(df_large, n_jobs_list=[1, 2, 4, -1], pre_window=10, post_window=15, ylags=10)
