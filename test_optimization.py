"""
Test script to verify the parallel processing optimization in LPDiD
"""

import numpy as np
import pandas as pd
import time
from LPDiD import LPDiD

def generate_test_data(n_units=1000, n_periods=20, seed=42):
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
    
    # Generate treatment (some units get treated at period 10)
    treated_units = np.random.choice(n_units, size=n_units//2, replace=False)
    df['treat'] = 0
    df.loc[(df['unit'].isin(treated_units)) & (df['time'] >= 10), 'treat'] = 1
    
    # Generate outcome with treatment effect
    df['y'] = np.random.randn(len(df)) + 0.1 * df['time']
    df.loc[df['treat'] == 1, 'y'] += 2.0  # Treatment effect
    
    return df

def test_optimization():
    """Test the optimization by comparing performance with different n_jobs settings"""
    print("Testing LPDiD Parallel Processing Optimization")
    print("=" * 50)
    
    # Generate test data
    df = generate_test_data(n_units=500, n_periods=20)
    print(f"Dataset: {len(df)} observations, {df['unit'].nunique()} units")
    
    # Test configurations
    test_configs = [
        {'n_jobs': 1, 'name': 'Sequential'},
        {'n_jobs': 2, 'name': '2 cores'},
        {'n_jobs': 4, 'name': '4 cores'},
        {'n_jobs': 8, 'name': '8 cores'},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\nTesting {config['name']} (n_jobs={config['n_jobs']})...")
        
        start_time = time.time()
        
        # Run LP-DiD
        lpdid = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=5,
            post_window=5,
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
            'n_estimates': len(result.event_study)
        }
        
        print(f"  Time: {elapsed_time:.2f} seconds")
        print(f"  Estimates: {len(result.event_study)} horizons")
    
    # Summary
    print("\n" + "=" * 50)
    print("Performance Summary")
    print("=" * 50)
    
    baseline_time = results['Sequential']['time']
    
    for name, data in results.items():
        speedup = baseline_time / data['time'] if data['time'] > 0 else float('inf')
        efficiency = speedup / data['n_jobs'] if data['n_jobs'] > 1 else 1.0
        
        print(f"{name:12s}: {data['time']:6.2f}s (speedup: {speedup:.2f}x, efficiency: {efficiency:.2f})")
    
    # Verify results are consistent
    print("\n" + "=" * 50)
    print("Verification: Checking result consistency")
    print("=" * 50)
    
    # Re-run with different configurations to verify consistency
    sequential_result = LPDiD(
        data=df, depvar='y', unit='unit', time='time', treat='treat',
        pre_window=3, post_window=3, formula='~ | time', n_jobs=1
    ).fit()
    
    parallel_result = LPDiD(
        data=df, depvar='y', unit='unit', time='time', treat='treat',
        pre_window=3, post_window=3, formula='~ | time', n_jobs=4
    ).fit()
    
    # Compare coefficients
    seq_coef = sequential_result.event_study['coefficient'].values
    par_coef = parallel_result.event_study['coefficient'].values
    
    max_diff = np.max(np.abs(seq_coef - par_coef))
    
    print(f"Maximum coefficient difference: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✓ Results are identical - optimization working correctly!")
    else:
        print("✗ Results differ - there may be an issue with the optimization")
    
    return results

if __name__ == "__main__":
    test_optimization()
