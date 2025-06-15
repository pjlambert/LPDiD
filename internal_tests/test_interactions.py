#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/pjl/Dropbox/LPDiD')

import numpy as np
import pandas as pd
from lpdid import LPDiD

# Generate test data with interaction effects
def create_interaction_test_data():
    np.random.seed(42)
    
    # Create a larger panel dataset
    n_units = 500
    n_periods = 35
    
    # Known treatment effects with STRONG interaction effects
    # Treatment effects by group type
    group_effects = {
        'low': {   # Effects for group = 'low'
            0: 5,   # Modest immediate effect for low group
            1: 6,   # Modest growth for low group
            2: 7,   # Modest growth for low group
            3: 8,   # Modest growth for low group
            4: 9,   # Modest growth for low group
            5: 10   # Modest growth for low group
        },
        'high': {  # Effects for group = 'high' (STRONG effects)
            0: 20,  # LARGE effect at horizon 0 for high group
            1: 24,  # LARGE effect at horizon 1 for high group
            2: 28,  # LARGE effect at horizon 2 for high group
            3: 32,  # LARGE effect at horizon 3 for high group
            4: 36,  # LARGE effect at horizon 4 for high group
            5: 40   # LARGE effect at horizon 5 for high group
        }
    }
    
    data = []
    for unit in range(1, n_units + 1):
        # Treatment timing staggered between periods 15-21 (more in the middle)
        treat_period = 15 + (unit % 7)
        
        # Create group indicator (categorical variable with 'low' and 'high')
        # Exactly half the units in each group
        group_type = 'high' if unit <= n_units // 2 else 'low'
        
        # Unit fixed effect
        unit_fe = np.random.normal(0, 2)
        
        for period in range(1, n_periods + 1):
            treated = 1 if period >= treat_period else 0
            periods_since_treatment = period - treat_period if treated else -1
            
            # Time fixed effect
            time_fe = np.random.normal(0, 1)
            
            # Control variable effect (keep this moderate)
            control_var = np.random.normal(0, 1)
            control_effect = 3.0 * control_var
            
            # Treatment effect based on group and time since treatment
            if treated and periods_since_treatment in group_effects[group_type]:
                treatment_effect = group_effects[group_type][periods_since_treatment]
            else:
                treatment_effect = 0
            
            # Base outcome with reduced noise to make effects more detectable
            outcome = (50 + unit_fe + time_fe + control_effect + 
                      treatment_effect + np.random.normal(0, 0.8))  # Reduced noise
            
            data.append({
                'unit_id': unit,
                'time_period': period,
                'outcome': outcome,
                'treated': treated,
                'control_var': control_var,
                'group_type': group_type,  # Single categorical variable
                'treat_period': treat_period,
                'periods_since_treatment': periods_since_treatment if treated else -99
            })
    
    df = pd.DataFrame(data)
    
    print("STRONG PRIORS FOR GROUP-SPECIFIC EFFECTS:")
    print("="*50)
    print("Effects by group:")
    for group, effects in group_effects.items():
        print(f"\n{group.upper()} group effects:")
        for horizon, effect in effects.items():
            print(f"  Horizon {horizon}: {effect}")
    
    print(f"\nControl variable coefficient should be around: 3.0")
    print(f"Treatment timing: periods {df['treat_period'].min()} to {df['treat_period'].max()}")
    print(f"High group units: {df[df['group_type']=='high']['unit_id'].nunique()}")
    print(f"Low group units: {df[df['group_type']=='low']['unit_id'].nunique()}")
    print(f"Reduced noise level for cleaner detection")
    
    # Calculate some summary statistics to verify our design
    treated_data = df[df['treated'] == 1]
    if len(treated_data) > 0:
        print(f"\nData validation:")
        print(f"Mean outcome for treated high group: {treated_data[treated_data['group_type']=='high']['outcome'].mean():.2f}")
        print(f"Mean outcome for treated low group: {treated_data[treated_data['group_type']=='low']['outcome'].mean():.2f}")
        print(f"Difference: {treated_data[treated_data['group_type']=='high']['outcome'].mean() - treated_data[treated_data['group_type']=='low']['outcome'].mean():.2f}")
    
    return df, group_effects

# Test interaction functionality with bias evaluation
def test_binary_interactions():
    print("="*60)
    print("TESTING BINARY INTERACTION TERMS - BIAS EVALUATION")
    print("="*60)
    
    print("\nCreating test data with STRONG interaction priors...")
    data, group_effects = create_interaction_test_data()
    print(f"Data shape: {data.shape}")
    
    print("\nRunning LP-DiD with categorical interaction (group_type)...")
    print("Using n_jobs=1 (no multiprocessing)")
    
    # Test with explicit formula specification to avoid multicollinearity
    lpdid = LPDiD(
        data=data,
        depvar='outcome',
        unit='unit_id', 
        time='time_period',
        treat='treated',
        pre_window=4,
        post_window=5,
        # Use explicit formula to control what's included
        formula="~ control_var | unit_id + time_period",
        interactions="~ group_type",  # Test categorical interaction
        n_jobs=1  # Use single processing as requested
    )
    
    print("\nDIAGNOSTIC: Checking data structure before estimation...")
    print(f"Unique values in group_type: {data['group_type'].unique()}")
    print(f"Group variation within units:")
    variation_check = data.groupby('unit_id')['group_type'].nunique()
    print(f"  Units with constant group_type: {(variation_check == 1).sum()}")
    print(f"  Units with varying group_type: {(variation_check > 1).sum()}")
    
    results = lpdid.fit()
    
    print("\nMain Results:")
    print(results.event_study[['horizon', 'coefficient', 'se', 't', 'p', 'obs']])
    
    # Check for group-specific coefficients
    group_cols = [col for col in results.event_study.columns if 'group_type_' in col and '_coef' in col]
    if group_cols:
        print("\nGroup-specific Results:")
        # Show all group-specific columns
        all_group_cols = [col for col in results.event_study.columns if 'group_type_' in col]
        print(results.event_study[['horizon'] + all_group_cols])
        
        print("\nBIAS EVALUATION - Comparison with TRUE STRONG effects:")
        print("="*90)
        print("Horizon | Low Group (low) | True Low | Bias | High Group (high) | True High | Bias")
        print("--------|-----------------|----------|------|-------------------|-----------|------")
        
        total_bias_low = 0
        total_bias_high = 0
        n_post_periods = 0
        
        for _, row in results.event_study.iterrows():
            horizon = int(row['horizon'])
            
            # Extract group-specific coefficients
            low_coef = row.get('group_type_low_coef', np.nan)
            high_coef = row.get('group_type_high_coef', np.nan)
            
            if horizon >= 0 and horizon in group_effects['low']:
                true_low = group_effects['low'][horizon]
                true_high = group_effects['high'][horizon]
                
                if not pd.isna(low_coef):
                    low_bias = low_coef - true_low
                    total_bias_low += abs(low_bias)
                else:
                    low_bias = np.nan
                
                if not pd.isna(high_coef):
                    high_bias = high_coef - true_high
                    total_bias_high += abs(high_bias)
                else:
                    high_bias = np.nan
                
                n_post_periods += 1
                
                print(f"{horizon:7} | {low_coef:15.2f} | {true_low:8.2f} | {low_bias:4.2f} | {high_coef:17.2f} | {true_high:9.2f} | {high_bias:4.2f}")
            else:
                # Pre-treatment periods should be close to zero
                print(f"{horizon:7} | {low_coef:15.2f} | {'0.00':>8} | {low_coef:4.2f} | {high_coef:17.2f} | {'0.00':>9} | {high_coef:4.2f}")
        
        if n_post_periods > 0:
            avg_low_bias = total_bias_low / n_post_periods
            avg_high_bias = total_bias_high / n_post_periods
            print(f"\nBIAS SUMMARY:")
            print(f"Average absolute bias in low group effects: {avg_low_bias:.3f}")
            print(f"Average absolute bias in high group effects: {avg_high_bias:.3f}")
            
            if avg_low_bias < 0.5 and avg_high_bias < 2.0:
                print("✅ LOW BIAS: Estimation appears unbiased")
            elif avg_low_bias < 1.0 and avg_high_bias < 4.0:
                print("⚠️  MODERATE BIAS: Some bias detected but acceptable")
            else:
                print("❌ HIGH BIAS: Significant bias detected!")
    else:
        print("\n❌ ERROR: No group-specific results found in output!")
        print("Available columns:", list(results.event_study.columns))
    
    return results

def main():
    print("Testing LP-DiD Interaction Terms Functionality (CATEGORICAL ONLY)")
    print("=" * 80)
    
    # Test 1: Binary interactions
    try:
        results1 = test_binary_interactions()
        print("\n✅ Binary interaction test completed")
    except Exception as e:
        print(f"\n❌ Binary interaction test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("INTERACTION TESTS SUMMARY (CATEGORICAL ONLY)")
    print("="*80)
    print("Binary/categorical interaction functionality test completed.")
    print("Check the output above to verify that:")
    print("1. Binary interactions are properly estimated")
    print("2. Interaction coefficients are close to true values")
    print("3. No multicollinearity issues are present")

if __name__ == "__main__":
    main()