"""
Test to verify that the vectorized approach produces identical results to a manual implementation
"""

import numpy as np
import pandas as pd
from LPDiD import LPDiD

def manual_long_differences(data, depvar, unit, time, post_window, pre_window, 
                           pmd=None, min_time_controls=False, swap_pre_diff=False):
    """
    Manual implementation of long differences to verify the vectorized approach
    """
    data = data.copy()
    data = data.sort_values([unit, time])
    
    # Handle pmd if specified
    if pmd == 'max':
        def calc_pre_mean(group):
            cumsum = group[depvar].expanding().sum()
            count = group[depvar].expanding().count()
            return cumsum.shift(1) / count.shift(1)
        
        data['aveLY'] = data.groupby(unit).apply(calc_pre_mean).reset_index(drop=True)
    elif isinstance(pmd, int):
        def calc_ma(group, window):
            return group[depvar].rolling(window=window, min_periods=window).mean()
        
        data['aveLY'] = data.groupby(unit).apply(
            lambda x: calc_ma(x, pmd).shift(1)
        ).reset_index(drop=True)
    
    # Generate differences manually
    for h in range(post_window + 1):
        col_name = f'D{h}y'
        if pmd is None:
            if min_time_controls:
                # For post-treatment periods (h >= 0), always use t-1 as control
                data[col_name] = (
                    data.groupby(unit)[depvar].shift(h) - 
                    data.groupby(unit)[depvar].shift(-1)
                )
            else:
                # Standard implementation
                data[col_name] = (
                    data.groupby(unit)[depvar].shift(h) - 
                    data.groupby(unit)[depvar].shift(-1)
                )
        else:
            data[col_name] = (
                data.groupby(unit)[depvar].shift(h) - 
                data['aveLY']
            )
    
    for h in range(2, pre_window + 1):
        col_name = f'Dm{h}y'
        if pmd is None:
            if min_time_controls:
                if h == 2:
                    control_shift = -1
                else:
                    control_shift = -(h - 1)
                
                if swap_pre_diff:
                    data[col_name] = (
                        data.groupby(unit)[depvar].shift(control_shift) - 
                        data.groupby(unit)[depvar].shift(-h)
                    )
                else:
                    data[col_name] = (
                        data.groupby(unit)[depvar].shift(-h) - 
                        data.groupby(unit)[depvar].shift(control_shift)
                    )
            else:
                if swap_pre_diff:
                    data[col_name] = (
                        data.groupby(unit)[depvar].shift(-1) - 
                        data.groupby(unit)[depvar].shift(-h)
                    )
                else:
                    data[col_name] = (
                        data.groupby(unit)[depvar].shift(-h) - 
                        data.groupby(unit)[depvar].shift(-1)
                    )
        else:
            if swap_pre_diff:
                data[col_name] = (
                    data['aveLY'] - 
                    data.groupby(unit)[depvar].shift(-h)
                )
            else:
                data[col_name] = (
                    data.groupby(unit)[depvar].shift(-h) - 
                    data['aveLY']
                )
    
    return data

def generate_test_data(n_units=100, n_periods=20, seed=42):
    """Generate test data for verification"""
    np.random.seed(seed)
    
    units = []
    periods = []
    for i in range(n_units):
        units.extend([i] * n_periods)
        periods.extend(list(range(n_periods)))
    
    df = pd.DataFrame({
        'unit': units,
        'time': periods
    })
    
    # Generate treatment
    treated_units = np.random.choice(n_units, size=n_units//2, replace=False)
    df['treat'] = 0
    df.loc[(df['unit'].isin(treated_units)) & (df['time'] >= 10), 'treat'] = 1
    
    # Generate outcome
    df['y'] = np.random.randn(len(df)) + 0.1 * df['time']
    df.loc[df['treat'] == 1, 'y'] += 2.0
    
    return df

def test_vectorization_correctness():
    """Test that vectorized and manual implementations produce identical results"""
    print("Testing Vectorization Correctness")
    print("=" * 50)
    
    # Generate test data
    df = generate_test_data(n_units=50, n_periods=15, seed=123)
    
    test_configs = [
        {
            'name': 'Standard case',
            'params': {'post_window': 3, 'pre_window': 4}
        },
        {
            'name': 'With min_time_controls', 
            'params': {'post_window': 3, 'pre_window': 4, 'min_time_controls': True}
        },
        {
            'name': 'With swap_pre_diff',
            'params': {'post_window': 3, 'pre_window': 4, 'swap_pre_diff': True}
        },
        {
            'name': 'With pmd=3',
            'params': {'post_window': 3, 'pre_window': 4, 'pmd': 3}
        },
        {
            'name': 'Complex case',
            'params': {'post_window': 3, 'pre_window': 4, 'min_time_controls': True, 'swap_pre_diff': True}
        }
    ]
    
    all_tests_passed = True
    
    for config in test_configs:
        print(f"\nTesting {config['name']}...")
        
        # Get manual implementation results
        manual_data = manual_long_differences(
            df, 'y', 'unit', 'time', **config['params']
        )
        
        # Get LPDiD vectorized results (force vectorized by using n_jobs=1)
        lpdid = LPDiD(
            data=df, depvar='y', unit='unit', time='time', treat='treat',
            formula='~ | time', n_jobs=1, **config['params']
        )
        
        # Extract the generated difference columns
        diff_cols = [col for col in lpdid.data.columns if col.startswith('D') and col.endswith('y')]
        
        max_diff = 0
        mismatched_cols = []
        
        for col in diff_cols:
            if col in manual_data.columns:
                # Compare the columns (ignoring NaN values)
                manual_vals = manual_data[col].dropna()
                vectorized_vals = lpdid.data[col].dropna()
                
                # Align the series for comparison
                common_idx = manual_vals.index.intersection(vectorized_vals.index)
                if len(common_idx) > 0:
                    manual_subset = manual_vals.loc[common_idx]
                    vectorized_subset = vectorized_vals.loc[common_idx]
                    
                    diff = np.max(np.abs(manual_subset - vectorized_subset))
                    max_diff = max(max_diff, diff)
                    
                    if diff > 1e-10:
                        mismatched_cols.append((col, diff))
        
        if max_diff < 1e-10:
            print(f"  ✓ PASSED - Maximum difference: {max_diff:.2e}")
        else:
            print(f"  ✗ FAILED - Maximum difference: {max_diff:.2e}")
            print(f"    Mismatched columns: {mismatched_cols}")
            all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Vectorization is working correctly!")
    else:
        print("❌ SOME TESTS FAILED! There are still issues with vectorization.")
    
    return all_tests_passed

if __name__ == "__main__":
    test_vectorization_correctness()
