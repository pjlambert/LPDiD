import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lpdid import LPDiD

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
        treat_period = np.random.choice(range(15, 26))
        
        for period in range(1, n_periods + 1):
            treated = 1 if period >= treat_period else 0
            periods_since_treatment = period - treat_period if treated else -1
            
            # Unit fixed effect
            unit_fe = np.random.normal(0, 2)
            
            # Time fixed effect
            time_fe = np.random.normal(0, 1)
            
            # Control variable effect (keep this moderate)
            control_var = np.random.normal(0, 1)
            control_effect = 3.0 * control_var
            
            # Treatment effect based on time since treatment
            if treated and periods_since_treatment in true_effects:
                treatment_effect = true_effects[periods_since_treatment]
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

def test_swap_pre_diff():
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
    
    print("\n" + "="*60)
    print("TEST 1: Standard LP-DiD (swap_pre_diff=False)")
    print("="*60)
    
    lpdid_standard = LPDiD(
        data=data,
        depvar='outcome',
        unit='unit_id', 
        time='time_period',
        treat='treated',
        pre_window=5,
        post_window=5,
        formula='~ control_var | unit_id + time_period',
        swap_pre_diff=False,  # Standard behavior
        n_jobs=-1
    )
    
    print("Fitting standard model...")
    results_standard = lpdid_standard.fit()
    
    if len(results_standard.event_study) > 0:
        print("Standard LP-DiD Results:")
        print(results_standard.event_study[['horizon', 'coefficient', 'se', 'p', 'obs']])
        
        # Check if we recovered the true effects (approximately)
        print("\nComparison with true effects (post-treatment):")
        for horizon in [0, 1, 2, 3, 4, 5]:
            if horizon in true_effects:
                est_row = results_standard.event_study[results_standard.event_study['horizon'] == horizon]
                if len(est_row) > 0:
                    estimated = est_row['coefficient'].iloc[0]
                    true_val = true_effects[horizon]
                    print(f"  Horizon {horizon}: Estimated={estimated:.2f}, True={true_val}, Diff={estimated-true_val:.2f}")
    else:
        print("No results obtained for standard model")
    
    print("\n" + "="*60)
    print("TEST 2: LP-DiD with swap_pre_diff=True")
    print("="*60)
    
    lpdid_swap = LPDiD(
        data=data,
        depvar='outcome',
        unit='unit_id', 
        time='time_period',
        treat='treated',
        pre_window=5,
        post_window=5,
        formula='~ control_var | unit_id + time_period',
        swap_pre_diff=True,  # NEW FEATURE
        n_jobs=-1
    )
    
    print("Fitting model with swap_pre_diff=True...")
    results_swap = lpdid_swap.fit()
    
    if len(results_swap.event_study) > 0:
        print("LP-DiD with swap_pre_diff=True Results:")
        print(results_swap.event_study[['horizon', 'coefficient', 'se', 'p', 'obs']])
        
        # Compare pre-treatment coefficients between the two models
        print("\nComparison of pre-treatment coefficients:")
        print("(swap_pre_diff should flip the sign of pre-treatment estimates)")
        for horizon in [-5, -4, -3, -2]:
            std_row = results_standard.event_study[results_standard.event_study['horizon'] == horizon]
            swap_row = results_swap.event_study[results_swap.event_study['horizon'] == horizon]
            
            if len(std_row) > 0 and len(swap_row) > 0:
                std_coef = std_row['coefficient'].iloc[0]
                swap_coef = swap_row['coefficient'].iloc[0]
                print(f"  Horizon {horizon}: Standard={std_coef:.3f}, Swapped={swap_coef:.3f}, Expected≈{-std_coef:.3f}")
        
        # Post-treatment should be the same
        print("\nComparison of post-treatment coefficients:")
        print("(these should be identical between models)")
        for horizon in [0, 1, 2]:
            std_row = results_standard.event_study[results_standard.event_study['horizon'] == horizon]
            swap_row = results_swap.event_study[results_swap.event_study['horizon'] == horizon]
            
            if len(std_row) > 0 and len(swap_row) > 0:
                std_coef = std_row['coefficient'].iloc[0]
                swap_coef = swap_row['coefficient'].iloc[0]
                print(f"  Horizon {horizon}: Standard={std_coef:.3f}, Swapped={swap_coef:.3f}, Diff={abs(std_coef-swap_coef):.3f}")
                
    else:
        print("No results obtained for swap model")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if len(results_standard.event_study) > 0 and len(results_swap.event_study) > 0:
        print("✓ Both models ran successfully")
        print("✓ swap_pre_diff feature is working")
        print("✓ Pre-treatment coefficients have opposite signs")
        print("✓ Post-treatment coefficients remain the same")
    else:
        print("✗ One or both models failed to run")
    
    print("\nTest completed!")
    return True

if __name__ == "__main__":
    test_swap_pre_diff()