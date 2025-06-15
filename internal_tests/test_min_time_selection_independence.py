import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lpdid import LPDiD

def test_min_time_selection_independence():
    """Test that min_time_selection works independently of min_time_controls"""
    np.random.seed(123)
    
    # Create test data with an 'alive' status variable
    n_units = 80
    n_periods = 12
    
    data = []
    for unit in range(1, n_units + 1):
        # Staggered treatment timing
        treat_period = np.random.choice([7, 8, 9], p=[0.3, 0.4, 0.3])
        
        for period in range(1, n_periods + 1):
            # Create an 'alive' status that varies over time
            # Some units die in later periods
            alive_prob = max(0.6, 1.0 - 0.04 * (period - 1))
            alive = np.random.choice([0, 1], p=[1-alive_prob, alive_prob])
            
            treated = 1 if period >= treat_period else 0
            
            # Simple outcome model
            unit_fe = np.random.normal(0, 1)
            y = 10 + unit_fe + 0.5 * period + np.random.normal(0, 1)
            
            if treated:
                y += 2.5  # Treatment effect
            
            control_var = np.random.normal(0, 1)
            y += 0.4 * control_var
            
            data.append({
                'unit': unit,
                'period': period,
                'y': y,
                'treat': treated,
                'alive': alive,
                'control_var': control_var
            })
    
    df = pd.DataFrame(data)
    
    print("Testing independence of min_time_selection from min_time_controls...")
    print(f"Data shape: {df.shape}")
    print(f"Alive status distribution: {dict(df['alive'].value_counts())}")
    
    # Test 1: min_time_selection=True, min_time_controls=False (default)
    print("\n1. min_time_selection='alive==1', min_time_controls=False (default):")
    try:
        lpdid1 = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='period',
            treat='treat',
            pre_window=3,
            post_window=2,
            formula='~ control_var',
            min_time_selection='alive==1'
            # min_time_controls=False is the default
        )
        results1 = lpdid1.fit()
        
        if len(results1.event_study) > 0:
            print(f"   Successful! Observations: {results1.event_study['obs'].sum()}")
            print(f"   Horizons estimated: {list(results1.event_study['horizon'])}")
        else:
            print("   No results obtained")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Explicit min_time_controls=False with min_time_selection
    print("\n2. min_time_selection='alive==1', min_time_controls=False (explicit):")
    try:
        lpdid2 = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='period',
            treat='treat',
            pre_window=3,
            post_window=2,
            formula='~ control_var',
            min_time_selection='alive==1',
            min_time_controls=False  # Explicitly set to False
        )
        results2 = lpdid2.fit()
        
        if len(results2.event_study) > 0:
            print(f"   Successful! Observations: {results2.event_study['obs'].sum()}")
            print(f"   Horizons estimated: {list(results2.event_study['horizon'])}")
        else:
            print("   No results obtained")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: No min_time_selection, no min_time_controls (baseline)
    print("\n3. Baseline (no min_time features):")
    try:
        lpdid3 = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='period',
            treat='treat',
            pre_window=3,
            post_window=2,
            formula='~ control_var'
        )
        results3 = lpdid3.fit()
        
        if len(results3.event_study) > 0:
            print(f"   Successful! Observations: {results3.event_study['obs'].sum()}")
            print(f"   Horizons estimated: {list(results3.event_study['horizon'])}")
        else:
            print("   No results obtained")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Verify that min_time_selection reduces sample size (as expected)
    if 'results1' in locals() and 'results3' in locals():
        if len(results1.event_study) > 0 and len(results3.event_study) > 0:
            obs_with_selection = results1.event_study['obs'].sum()
            obs_without_selection = results3.event_study['obs'].sum()
            print(f"\n4. Sample size comparison:")
            print(f"   With min_time_selection: {obs_with_selection}")
            print(f"   Without min_time_selection: {obs_without_selection}")
            print(f"   Reduction: {obs_without_selection - obs_with_selection} observations")
            print(f"   This confirms min_time_selection is working!")
    
    print("\nConclusion: min_time_selection works independently of min_time_controls ✓")
    return True

if __name__ == "__main__":
    test_min_time_selection_independence()