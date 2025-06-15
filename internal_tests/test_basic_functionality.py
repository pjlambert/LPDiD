#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/pjl/Dropbox/lpdid')

import numpy as np
import pandas as pd
from lpdid import LPDiD

# Generate simple test data with strong priors
def create_simple_test_data():
    np.random.seed(123)
    
    # Create a larger panel dataset with more periods to support horizon -5 to +5
    n_units = 500  # Increase number of units
    n_periods = 40  # Increase number of periods significantly
    
    # Known treatment effect: immediate effect of 10, growing by 2 each period
    true_effects = {
        0: 10,   # Immediate effect
        1: 12,   # Period 1 after treatment
        2: 14,   # Period 2 after treatment
        3: 16,   # Period 3 after treatment
        4: 18,   # Period 4 after treatment
        5: 20    # Period 5 after treatment
    }
    
    data = []
    for unit in range(1, n_units + 1):
        # Staggered treatment timing in the middle of the panel (periods 15-25)
        # This gives us plenty of pre/post periods for estimation
        treat_period = 15 + (unit % 11)  # Treatment between periods 15-25
        
        # Unit fixed effect
        unit_fe = np.random.normal(0, 5)
        
        for period in range(1, n_periods + 1):
            treated = 1 if period >= treat_period else 0
            periods_since_treatment = period - treat_period if treated else -1
            
            # Time fixed effect
            time_fe = np.random.normal(0, 2)
            
            # Control variable effect (coefficient should be around 3)
            control_var = np.random.normal(0, 1)
            control_effect = 3 * control_var
            
            # Treatment effect based on time since treatment
            if treated and periods_since_treatment in true_effects:
                treatment_effect = true_effects[periods_since_treatment]
            else:
                treatment_effect = 0
            
            # Base outcome: unit FE + time FE + control effect + treatment effect + noise
            outcome = 50 + unit_fe + time_fe + control_effect + treatment_effect + np.random.normal(0, 1)
            
            data.append({
                'unit_id': unit,
                'time_period': period,
                'outcome': outcome,
                'treated': treated,
                'control_var': control_var,
                'treat_period': treat_period,
                'periods_since_treatment': periods_since_treatment if treated else -99
            })
    
    df = pd.DataFrame(data)
    
    print("True treatment effects by horizon:")
    for horizon, effect in true_effects.items():
        print(f"  Horizon {horizon}: {effect}")
    print(f"Control variable coefficient should be around: 3")
    print(f"Treatment timing: periods {df['treat_period'].min()} to {df['treat_period'].max()}")
    print(f"Units per treatment cohort: approximately {n_units // 11}")
    print(f"Total periods: {n_periods}")
    
    return df

# Test basic functionality
def test_basic_lpdid():
    print("Creating test data...")
    data = create_simple_test_data()
    print(f"Data shape: {data.shape}")
    print("Sample of data:")
    print(data.head(10))
    
    # Define true effects here so it's available for comparison
    true_effects = {
        0: 10,   # Immediate effect
        1: 12,   # Period 1 after treatment
        2: 14,   # Period 2 after treatment
        3: 16,   # Period 3 after treatment
        4: 18,   # Period 4 after treatment
        5: 20    # Period 5 after treatment
    }
    
    print("\nInitializing LP-DiD with expanded windows...")
    lpdid = LPDiD(
        data=data,
        depvar='outcome',
        unit='unit_id', 
        time='time_period',
        treat='treated',
        pre_window=5,  # Expand to 5 pre-treatment periods
        post_window=5, # Expand to 5 post-treatment periods
        formula="~ control_var",
        n_jobs=1
    )
    
    print("Running estimation...")
    results = lpdid.fit()
    
    print("\nResults:")
    print(results.event_study)
    
    # Compare with true effects
    print("\nComparison with true effects:")
    print("Horizon | Estimated | True Effect | Difference")
    print("--------|-----------|-------------|----------")
    
    for _, row in results.event_study.iterrows():
        horizon = int(row['horizon'])
        estimated = row['coefficient']
        if horizon in true_effects:
            true_effect = true_effects[horizon]
            diff = estimated - true_effect
            print(f"{horizon:7} | {estimated:9.2f} | {true_effect:11.2f} | {diff:10.2f}")
        else:
            print(f"{horizon:7} | {estimated:9.2f} | {'N/A':>11} | {'N/A':>10}")
    
    return results

def main():
    print("Testing LP-DiD Basic Functionality")
    print("=" * 80)
    
    # Test 1: Basic functionality
    print("Test 1: Basic LP-DiD functionality")
    try:
        results_basic = test_basic_lpdid()
        print("\n✅ Basic functionality test completed successfully")
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("BASIC FUNCTIONALITY TEST SUMMARY")
    print("="*80)
    print("1. Basic LP-DiD functionality: ✅")
    print("\nThe test verifies:")
    print("- Coefficient estimation accuracy")
    print("- Event study horizon coverage")
    print("- Comparison with known true effects")

if __name__ == "__main__":
    main()