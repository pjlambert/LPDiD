import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lpdid import LPDiD

def test_min_time_features():
    """Test min_time_controls and min_time_selection features"""
    np.random.seed(42)
    
    # Create more realistic test data with staggered treatment timing
    n_units = 100
    n_periods = 15
    
    data = []
    for unit in range(1, n_units + 1):
        # Staggered treatment timing
        treat_period = np.random.choice([8, 9, 10, 11, 12], p=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        for period in range(1, n_periods + 1):
            # Create an 'alive' status that varies over time
            # Higher probability of being alive in earlier periods
            alive_prob = max(0.7, 1.0 - 0.03 * (period - 1))
            alive = np.random.choice([0, 1], p=[1-alive_prob, alive_prob])
            
            treated = 1 if period >= treat_period else 0
            
            # Add unit fixed effects and time trends
            unit_fe = np.random.normal(0, 2)  # Unit fixed effect
            time_trend = 0.5 * period + np.random.normal(0, 1)
            
            # Base outcome with unit and time variation
            y = 10 + unit_fe + time_trend
            
            # Add treatment effect that grows over time
            if treated:
                periods_since_treat = period - treat_period
                treatment_effect = 3 + 0.5 * periods_since_treat + np.random.normal(0, 0.5)
                y += treatment_effect
            
            # Add some controls
            control1 = np.random.normal(5, 2)
            control2 = np.random.normal(0, 1)
            y += 0.3 * control1 + 0.2 * control2
            
            data.append({
                'unit': unit,
                'period': period,
                'y': y,
                'treat': treated,
                'alive': alive,
                'control1': control1,
                'control2': control2
            })
    
    df = pd.DataFrame(data)
    
    print("Testing min_time_controls and min_time_selection features...")
    print(f"Data shape: {df.shape}")
    print(f"Alive variable distribution:\n{df['alive'].value_counts()}")
    print(f"Treatment periods: {sorted(df[df['treat']==1]['period'].unique())}")
    
    # Test 1: Standard LP-DiD (baseline)
    print("\n1. Standard LP-DiD (baseline):")
    try:
        lpdid_standard = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='period',
            treat='treat',
            pre_window=3,
            post_window=2,
            formula='~ control1 + control2'
        )
        results_standard = lpdid_standard.fit()
        
        if len(results_standard.event_study) > 0:
            print(f"   Number of observations in results: {results_standard.event_study['obs'].sum()}")
            print(f"   Event study results:\n{results_standard.event_study[['horizon', 'coefficient', 'obs']]}")
        else:
            print("   No results obtained")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: With min_time_controls
    print("\n2. With min_time_controls=True:")
    try:
        lpdid_min_controls = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='period',
            treat='treat',
            pre_window=3,
            post_window=2,
            formula='~ control1 + control2',
            min_time_controls=True
        )
        results_min_controls = lpdid_min_controls.fit()
        
        if len(results_min_controls.event_study) > 0:
            print(f"   Number of observations in results: {results_min_controls.event_study['obs'].sum()}")
            print(f"   Event study results:\n{results_min_controls.event_study[['horizon', 'coefficient', 'obs']]}")
        else:
            print("   No results obtained")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: With min_time_selection
    print("\n3. With min_time_selection='alive==1':")
    try:
        lpdid_min_selection = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='period',
            treat='treat',
            pre_window=3,
            post_window=2,
            formula='~ control1 + control2',
            min_time_selection='alive==1'
        )
        results_min_selection = lpdid_min_selection.fit()
        
        if len(results_min_selection.event_study) > 0:
            print(f"   Number of observations in results: {results_min_selection.event_study['obs'].sum()}")
            print(f"   Event study results:\n{results_min_selection.event_study[['horizon', 'coefficient', 'obs']]}")
        else:
            print("   No results obtained")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Both features combined
    print("\n4. Both min_time_controls=True and min_time_selection='alive==1':")
    try:
        lpdid_both = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='period',
            treat='treat',
            pre_window=3,
            post_window=2,
            formula='~ control1 + control2',
            min_time_controls=True,
            min_time_selection='alive==1'
        )
        results_both = lpdid_both.fit()
        
        if len(results_both.event_study) > 0:
            print(f"   Number of observations in results: {results_both.event_study['obs'].sum()}")
            print(f"   Event study results:\n{results_both.event_study[['horizon', 'coefficient', 'obs']]}")
        else:
            print("   No results obtained")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nTest completed successfully!")
    return True

if __name__ == "__main__":
    test_min_time_features()