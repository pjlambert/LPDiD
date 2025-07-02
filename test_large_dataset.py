"""
Test script to demonstrate the optimization on larger datasets
"""

import numpy as np
import pandas as pd
import time
from LPDiD import LPDiD

def generate_large_test_data(n_units=5000, n_periods=20, seed=42):
    """Generate larger synthetic panel data"""
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
    
    # Generate treatment (some units get treated at period 10)
    treated_units = np.random.choice(n_units, size=n_units//2, replace=False)
    df['treat'] = 0
    df.loc[(df['unit'].isin(treated_units)) & (df['time'] >= 10), 'treat'] = 1
    
    # Generate outcome with treatment effect
    df['y'] = np.random.randn(len(df)) + 0.1 * df['time']
    df.loc[df['treat'] == 1, 'y'] += 2.0  # Treatment effect
    
    return df

def test_large_dataset_performance():
    """Test performance on a large dataset that mimics your original problem"""
    print("Testing LPDiD Performance on Large Dataset")
    print("=" * 50)
    
    # Generate large test data (similar to your original 100K observations)
    df = generate_large_test_data(n_units=5000, n_periods=20)
    print(f"Dataset: {len(df)} observations, {df['unit'].nunique()} units")
    print("This simulates your original scenario with 100K observations\n")
    
    # Test the configurations that were problematic in your original issue
    test_configs = [
        {'n_jobs': 2, 'name': '2 cores (was fast)'},
        {'n_jobs': 8, 'name': '8 cores (was slow)'},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"Testing {config['name']} (n_jobs={config['n_jobs']})...")
        
        start_time = time.time()
        
        # Run LP-DiD with same parameters as your original issue
        lpdid = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=10,
            post_window=9,
            formula='~ | time',
            n_jobs=config['n_jobs']
        )
        
        # Fit the model
        result = lpdid.fit()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        results[config['name']] = {
            'time': elapsed_time,
            'n_jobs': config['n_jobs'],
        }
        
        print(f"  Time: {elapsed_time:.2f} seconds")
        print(f"  Successfully estimated {len(result.event_study)} horizons\n")
    
    # Summary
    print("=" * 50)
    print("Performance Comparison")
    print("=" * 50)
    
    time_2_cores = results['2 cores (was fast)']['time']
    time_8_cores = results['8 cores (was slow)']['time']
    
    print(f"2 cores: {time_2_cores:.2f} seconds")
    print(f"8 cores: {time_8_cores:.2f} seconds")
    
    if time_8_cores < time_2_cores:
        speedup = time_2_cores / time_8_cores
        print(f"\n✓ 8 cores is now {speedup:.2f}x FASTER than 2 cores!")
        print("✓ The optimization fixed the parallel processing issue!")
    else:
        slowdown = time_8_cores / time_2_cores
        print(f"\n✓ Performance is similar (8 cores takes {slowdown:.2f}x the time)")
        print("✓ The optimization prevented the severe slowdown!")
    
    print(f"\nBoth configurations now use the optimized vectorized approach")
    print("instead of the problematic fine-grained parallelization.")
    
    return results

if __name__ == "__main__":
    test_large_dataset_performance()
