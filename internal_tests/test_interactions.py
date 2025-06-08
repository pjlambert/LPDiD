#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/pjl/Dropbox/lpdid')

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
    # Base treatment effect (for low_group = 0)
    base_treatment_effects = {
        0: 5,   # Modest immediate effect for low group
        1: 6,   # Modest growth for low group
        2: 7,   # Modest growth for low group
        3: 8,   # Modest growth for low group
        4: 9,   # Modest growth for low group
        5: 10   # Modest growth for low group
    }
    
    # STRONG interaction effects (additional effect for high_group = 1)
    # These should be large and easily detectable
    interaction_effects = {
        0: 15,  # LARGE additional effect at horizon 0 for high_group
        1: 18,  # LARGE additional effect at horizon 1 for high_group
        2: 21,  # LARGE additional effect at horizon 2 for high_group
        3: 24,  # LARGE additional effect at horizon 3 for high_group
        4: 27,  # LARGE additional effect at horizon 4 for high_group
        5: 30   # LARGE additional effect at horizon 5 for high_group
    }
    
    # Strong continuous interaction effect (COMMENTED OUT FOR NOW)
    # continuous_interaction_coef = 8.0  # Large coefficient for continuous interaction
    
    data = []
    for unit in range(1, n_units + 1):
        # Treatment timing staggered between periods 15-21 (more in the middle)
        treat_period = 15 + (unit % 7)
        
        # Create subgroup indicator (high_group = 1 for exactly half the units)
        high_group = 1 if unit <= n_units // 2 else 0
        
        # Create continuous interaction variable with more variation (COMMENTED OUT FOR NOW)
        # continuous_var = np.random.normal(0, 1.5)  # Increased variance
        
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
            
            # Base treatment effect
            if treated and periods_since_treatment in base_treatment_effects:
                main_treatment_effect = base_treatment_effects[periods_since_treatment]
                
                # Add STRONG binary interaction effect for high_group
                if high_group == 1 and periods_since_treatment in interaction_effects:
                    binary_interaction_effect = interaction_effects[periods_since_treatment]
                else:
                    binary_interaction_effect = 0
                
                total_treatment_effect = main_treatment_effect + binary_interaction_effect
            else:
                total_treatment_effect = 0
            
            # STRONG continuous interaction effect (COMMENTED OUT FOR NOW)
            # continuous_interaction_effect = continuous_interaction_coef * continuous_var * treated if treated else 0
            
            # Base outcome with reduced noise to make effects more detectable
            outcome = (50 + unit_fe + time_fe + control_effect + 
                      total_treatment_effect + # continuous_interaction_effect + 
                      np.random.normal(0, 0.8))  # Reduced noise
            
            data.append({
                'unit_id': unit,
                'time_period': period,
                'outcome': outcome,
                'treated': treated,
                'control_var': control_var,
                'high_group': high_group,
                # 'continuous_var': continuous_var,  # COMMENTED OUT FOR NOW
                'treat_period': treat_period,
                'periods_since_treatment': periods_since_treatment if treated else -99
            })
    
    df = pd.DataFrame(data)
    
    print("STRONG PRIORS FOR INTERACTION EFFECTS:")
    print("="*50)
    print("Base effects (low_group = 0):")
    for horizon, effect in base_treatment_effects.items():
        print(f"  Horizon {horizon}: {effect}")
    
    print("\nSTRONG Binary interaction effects (additional for high_group = 1):")
    for horizon, effect in interaction_effects.items():
        print(f"  Horizon {horizon}: +{effect} (LARGE)")
    
    print("\nTotal effects for high_group:")
    for horizon in base_treatment_effects.keys():
        total = base_treatment_effects[horizon] + interaction_effects[horizon]
        print(f"  Horizon {horizon}: {total} (low: {base_treatment_effects[horizon]}, high: {total})")
    
    # print(f"\nSTRONG Continuous interaction coefficient: {continuous_interaction_coef}")  # COMMENTED OUT
    print(f"\nControl variable coefficient should be around: 3.0")
    print(f"Treatment timing: periods {df['treat_period'].min()} to {df['treat_period'].max()}")
    print(f"High group units: {df[df['high_group']==1]['unit_id'].nunique()}")
    print(f"Low group units: {df[df['high_group']==0]['unit_id'].nunique()}")
    print(f"Reduced noise level for cleaner detection")
    
    # Calculate some summary statistics to verify our design
    treated_data = df[df['treated'] == 1]
    if len(treated_data) > 0:
        print(f"\nData validation:")
        print(f"Mean outcome for treated high_group: {treated_data[treated_data['high_group']==1]['outcome'].mean():.2f}")
        print(f"Mean outcome for treated low_group: {treated_data[treated_data['high_group']==0]['outcome'].mean():.2f}")
        print(f"Difference: {treated_data[treated_data['high_group']==1]['outcome'].mean() - treated_data[treated_data['high_group']==0]['outcome'].mean():.2f}")
    
    return df, base_treatment_effects, interaction_effects  # , continuous_interaction_coef  # COMMENTED OUT

# Test interaction functionality with bias evaluation
def test_binary_interactions():
    print("="*60)
    print("TESTING BINARY INTERACTION TERMS - BIAS EVALUATION")
    print("="*60)
    
    print("\nCreating test data with STRONG interaction priors...")
    data, base_effects, interaction_effects = create_interaction_test_data()  # Removed continuous_coef
    print(f"Data shape: {data.shape}")
    
    print("\nRunning LP-DiD with binary interaction (high_group)...")
    print("Testing WITHOUT time-varying fixed effects first...")
    
    # First try without time-varying fixed effects to see if the syntax issue is resolved
    lpdid = LPDiD(
        data=data,
        depvar='outcome',
        unit='unit_id', 
        time='time_period',
        treat='treated',
        pre_window=4,
        post_window=5,
        # Simplified formula without time-varying fixed effects for now
        formula="~ control_var | unit_id + time_period",
        interactions="~ high_group",  # Test binary interaction
        n_jobs=1
    )
    
    results = lpdid.fit()
    
    print("\nMain Results:")
    print(results.event_study[['horizon', 'coefficient', 'se', 't', 'p', 'obs']])
    
    # Check if interaction columns exist
    interaction_cols = [col for col in results.event_study.columns if 'interact_high_group' in col]
    if interaction_cols:
        print("\nInteraction Results (high_group):")
        print(results.event_study[['horizon'] + interaction_cols])
        
        print("\nBIAS EVALUATION - Comparison with TRUE STRONG effects:")
        print("="*80)
        print("Horizon | Main Coef | True Main | Bias | Interact Coef | True Interact | Bias | Total Effect")
        print("--------|-----------|-----------|------|---------------|---------------|------|-------------")
        
        total_bias_main = 0
        total_bias_interact = 0
        n_post_periods = 0
        
        for _, row in results.event_study.iterrows():
            horizon = int(row['horizon'])
            main_coef = row['coefficient']
            
            if horizon >= 0 and horizon in base_effects:
                true_main = base_effects[horizon]
                main_bias = main_coef - true_main
                total_bias_main += abs(main_bias)
                n_post_periods += 1
                
                if 'interact_high_group' in row and not pd.isna(row['interact_high_group']):
                    interact_coef = row['interact_high_group']
                    true_interact = interaction_effects.get(horizon, 0)
                    interact_bias = interact_coef - true_interact
                    total_bias_interact += abs(interact_bias)
                    
                    # Calculate implied total effect for high group
                    total_effect_est = main_coef + interact_coef
                    total_effect_true = true_main + true_interact
                    
                    print(f"{horizon:7} | {main_coef:9.2f} | {true_main:9.2f} | {main_bias:4.2f} | {interact_coef:13.2f} | {true_interact:13.2f} | {interact_bias:4.2f} | Est:{total_effect_est:.1f} True:{total_effect_true:.1f}")
                else:
                    print(f"{horizon:7} | {main_coef:9.2f} | {true_main:9.2f} | {main_bias:4.2f} | {'N/A':>13} | {'N/A':>13} | {'N/A':>4} | {'N/A'}")
            else:
                # Pre-treatment periods should be close to zero
                print(f"{horizon:7} | {main_coef:9.2f} | {'0.00':>9} | {main_coef:4.2f} | {'N/A':>13} | {'N/A':>13} | {'N/A':>4} | Pre-treatment")
        
        if n_post_periods > 0:
            avg_main_bias = total_bias_main / n_post_periods
            avg_interact_bias = total_bias_interact / n_post_periods
            print(f"\nBIAS SUMMARY:")
            print(f"Average absolute bias in main effects: {avg_main_bias:.3f}")
            print(f"Average absolute bias in interaction effects: {avg_interact_bias:.3f}")
            
            if avg_main_bias < 0.5 and avg_interact_bias < 2.0:
                print("✅ LOW BIAS: Estimation appears unbiased")
            elif avg_main_bias < 1.0 and avg_interact_bias < 4.0:
                print("⚠️  MODERATE BIAS: Some bias detected but acceptable")
            else:
                print("❌ HIGH BIAS: Significant bias detected!")
    else:
        print("\n❌ ERROR: No interaction results found in output!")
    
    return results

# COMMENTED OUT FOR NOW - CONTINUOUS INTERACTIONS
# def test_continuous_interactions():
#     print("\n" + "="*60)
#     print("TESTING CONTINUOUS INTERACTION TERMS - BIAS EVALUATION")
#     print("="*60)
#     
#     print("\nCreating test data with STRONG continuous interaction...")
#     data, _, _, true_continuous_coef = create_interaction_test_data()
#     
#     print("\nRunning LP-DiD with continuous interaction (continuous_var)...")
#     print("Including continuous_var as time-varying fixed effect (continuous_var:time_period)...")
#     
#     lpdid = LPDiD(
#         data=data,
#         depvar='outcome',
#         unit='unit_id', 
#         time='time_period',
#         treat='treated',
#         pre_window=4,
#         post_window=5,
#         # Include continuous variable as both control and time-varying fixed effect
#         formula="~ control_var + continuous_var | unit_id + time_period + continuous_var:time_period",
#         interactions="~ continuous_var",  # Test continuous interaction
#         n_jobs=1
#     )
#     
#     results = lpdid.fit()
#     
#     print("\nMain Results:")
#     print(results.event_study[['horizon', 'coefficient', 'se', 't', 'p', 'obs']])
#     
#     # Check if interaction columns exist
#     interaction_cols = [col for col in results.event_study.columns if 'interact_continuous_var' in col]
#     if interaction_cols:
#         print("\nContinuous Interaction Results:")
#         interaction_data = results.event_study[['horizon'] + interaction_cols]
#         print(interaction_data)
#         
#         print(f"\nBIAS EVALUATION for Continuous Interaction:")
#         print(f"True continuous interaction coefficient: {true_continuous_coef}")
#         print("="*50)
#         
#         total_bias = 0
#         n_post_periods = 0
#         
#         for _, row in results.event_study.iterrows():
#             horizon = int(row['horizon'])
#             if horizon >= 0 and 'interact_continuous_var' in row and not pd.isna(row['interact_continuous_var']):
#                 est_coef = row['interact_continuous_var']
#                 bias = est_coef - true_continuous_coef
#                 total_bias += abs(bias)
#                 n_post_periods += 1
#                 print(f"Horizon {horizon}: Estimated={est_coef:.3f}, True={true_continuous_coef:.3f}, Bias={bias:.3f}")
#         
#         if n_post_periods > 0:
#             avg_bias = total_bias / n_post_periods
#             print(f"\nAverage absolute bias: {avg_bias:.3f}")
#             
#             if avg_bias < 0.5:
#                 print("✅ LOW BIAS: Continuous interaction estimation appears unbiased")
#             elif avg_bias < 1.0:
#                 print("⚠️  MODERATE BIAS: Some bias in continuous interaction")
#             else:
#                 print("❌ HIGH BIAS: Significant bias in continuous interaction!")
#     else:
#         print("\n❌ ERROR: No continuous interaction results found!")
#     
#     return results

# COMMENTED OUT FOR NOW - MULTIPLE INTERACTIONS
# def test_multiple_interactions():
#     print("\n" + "="*60)
#     print("TESTING MULTIPLE INTERACTION TERMS")
#     print("="*60)
#     
#     print("\nCreating test data...")
#     data, _, _, _ = create_interaction_test_data()
#     
#     print("\nRunning LP-DiD with multiple interactions...")
#     print("Including both high_group and continuous_var as time-varying fixed effects...")
#     
#     lpdid = LPDiD(
#         data=data,
#         depvar='outcome',
#         unit='unit_id', 
#         time='time_period',
#         treat='treated',
#         pre_window=3,
#         post_window=5,
#         # Include both interaction variables as time-varying fixed effects
#         formula="~ control_var + continuous_var | unit_id + time_period + high_group:time_period + continuous_var:time_period",
#         interactions="~ high_group + continuous_var",  # Test both interactions
#         n_jobs=1
#     )
#     
#     results = lpdid.fit()
#     
#     print("\nMain Results:")
#     print(results.event_study[['horizon', 'coefficient', 'se', 't', 'p', 'obs']])
#     
#     # Check for binary interaction results
#     binary_cols = [col for col in results.event_study.columns if 'interact_high_group' in col]
#     if binary_cols:
#         print("\nBinary Interaction Results (high_group):")
#         print(results.event_study[['horizon'] + binary_cols])
#     
#     # Check for continuous interaction results
#     continuous_cols = [col for col in results.event_study.columns if 'interact_continuous_var' in col]
#     if continuous_cols:
#         print("\nContinuous Interaction Results (continuous_var):")
#         print(results.event_study[['horizon'] + continuous_cols])
#     
#     return results

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
    
    # COMMENTED OUT FOR NOW
    # # Test 2: Continuous interactions  
    # try:
    #     results2 = test_continuous_interactions()
    #     print("\n✅ Continuous interaction test completed")
    # except Exception as e:
    #     print(f"\n❌ Continuous interaction test failed: {e}")
    # 
    # # Test 3: Multiple interactions
    # try:
    #     results3 = test_multiple_interactions()
    #     print("\n✅ Multiple interactions test completed")
    # except Exception as e:
    #     print(f"\n❌ Multiple interactions test failed: {e}")
    
    print("\n" + "="*80)
    print("INTERACTION TESTS SUMMARY (CATEGORICAL ONLY)")
    print("="*80)
    print("Binary/categorical interaction functionality test completed.")
    print("Check the output above to verify that:")
    print("1. Binary interactions are properly estimated")
    # print("2. Continuous interactions work correctly")  # COMMENTED OUT
    # print("3. Multiple interactions can be specified together")  # COMMENTED OUT
    print("2. Interaction coefficients are close to true values")

if __name__ == "__main__":
    main()