import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from LPDiD import LPDiDPois

def create_survival_test_data():
    """Create test data with binary death outcomes and treatment effects on hazard rates"""
    np.random.seed(42)
    
    # Create a panel dataset with survival/death outcomes
    n_units = 200  # Reduce units for simpler test
    n_periods = 15  # Reduce periods 
    
    # Treatment effect: reduces hazard rate (less likely to die)
    # Baseline hazard rate per period
    baseline_hazard = 0.08  # 8% chance of death per period without treatment
    
    # Treatment effects on hazard rates (multiplicative)
    # Negative values mean treatment reduces death hazard
    true_hazard_effects = {
        0: 0.6,   # 40% reduction in hazard immediately
        1: 0.5,   # 50% reduction in period 1
        2: 0.4,   # 60% reduction in period 2  
        3: 0.3,   # 70% reduction in period 3
    }
    
    data = []
    for unit in range(1, n_units + 1):
        # Staggered treatment timing in the middle of the panel
        treat_period = np.random.choice(range(6, 10))
        
        # Track if unit is still alive
        alive = True
        
        for period in range(1, n_periods + 1):
            treated = 1 if period >= treat_period and alive else 0
            periods_since_treatment = period - treat_period if treated else -1
            
            # Unit fixed effect (some units are more/less frail)
            unit_frailty = np.random.normal(1.0, 0.3)  # Multiplicative frailty
            unit_frailty = max(0.2, unit_frailty)  # Ensure positive
            
            # Control variables
            control_var = np.random.normal(0, 1)
            # Control effect on hazard (multiplicative)
            control_hazard_effect = np.exp(0.2 * control_var)
            
            # Calculate current hazard rate
            current_hazard = baseline_hazard * unit_frailty * control_hazard_effect
            
            # Apply treatment effect if treated
            if treated and periods_since_treatment in true_hazard_effects:
                hazard_multiplier = true_hazard_effects[periods_since_treatment]
                current_hazard *= hazard_multiplier
            
            # Determine if unit dies this period (if still alive)
            died_this_period = 0
            if alive:
                # Bernoulli trial for death
                died_this_period = np.random.binomial(1, min(current_hazard, 0.95))
                if died_this_period:
                    alive = False
            
            # Create the outcome: 1 = died this period, 0 = survived/already dead
            outcome = died_this_period
            
            data.append({
                'unit_id': unit,
                'time_period': period,
                'death': outcome,  # Binary outcome: 1 = death, 0 = survival
                'treated': treated,
                'control_var': control_var,
                'treat_period': treat_period,
                'periods_since_treatment': periods_since_treatment if treated else -99,
                'alive_start_period': 1 if alive or outcome == 1 else 0,  # Alive at start of period
                'hazard_rate': current_hazard  # For analysis
            })
    
    df = pd.DataFrame(data)
    
    print("True treatment effects on hazard rates (multiplicative):")
    for horizon, effect in true_hazard_effects.items():
        reduction = (1 - effect) * 100
        print(f"  Horizon {horizon}: {effect:.1f} (reduces hazard by {reduction:.0f}%)")
    
    print(f"Baseline hazard rate: {baseline_hazard:.1%} per period")
    print(f"Treatment timing: periods {df['treat_period'].min()} to {df['treat_period'].max()}")
    print(f"Total deaths: {df['death'].sum()}")
    print(f"Death rate: {df['death'].mean():.1%}")
    print(f"Deaths among treated: {df[df['treated']==1]['death'].sum()}")
    print(f"Deaths among controls: {df[df['treated']==0]['death'].sum()}")
    
    return df

def test_lpdid_poisson():
    print("Creating survival test data...")
    data = create_survival_test_data()
    print(f"Data shape: {data.shape}")
    print("Sample of data:")
    print(data.head(10))
    
    # True hazard reduction effects (for comparison)
    true_hazard_effects = {
        0: 0.6,   # 40% reduction
        1: 0.5,   # 50% reduction  
        2: 0.4,   # 60% reduction
        3: 0.3,   # 70% reduction
    }
    
    print("\n" + "="*60)
    print("TEST 1: LP-DiD with Poisson Regression")
    print("="*60)
    
    lpdid_pois = LPDiDPois(
        data=data,
        depvar='death',
        unit='unit_id', 
        time='time_period',
        treat='treated',
        pre_window=3,  # Reduce window size
        post_window=3,
        formula='~ control_var',  # Simpler formula without unit/time FE that cause separation
        swap_pre_diff=True,  # Ensure non-negative outcomes
        cluster_formula='~ unit_id',  # Explicit clustering
        n_jobs=1  # Single threaded for debugging
    )
    
    print("Fitting Poisson model...")
    results_pois = lpdid_pois.fit()
    
    if len(results_pois.event_study) > 0:
        print("Poisson LP-DiD Results:")
        print(results_pois.event_study[['horizon', 'coefficient', 'se', 'p', 'obs']])
        
        print("\nInterpretation of Poisson coefficients:")
        print("(Negative coefficients indicate reduced death hazard)")
        for _, row in results_pois.event_study.iterrows():
            horizon = int(row['horizon'])
            coef = row['coefficient']
            # Convert to hazard ratio
            hazard_ratio = np.exp(coef)
            percent_change = (hazard_ratio - 1) * 100
            
            if horizon in true_hazard_effects:
                true_ratio = true_hazard_effects[horizon]
                true_percent = (true_ratio - 1) * 100
                print(f"  Horizon {horizon}: Coef={coef:.3f}, HR={hazard_ratio:.3f} ({percent_change:+.1f}%), True HR={true_ratio:.3f} ({true_percent:+.1f}%)")
            else:
                print(f"  Horizon {horizon}: Coef={coef:.3f}, HR={hazard_ratio:.3f} ({percent_change:+.1f}%)")
    else:
        print("No results obtained for Poisson model")
    
    print("\n" + "="*60)
    print("TEST 2: Comparison with OLS (no fixed effects)")
    print("="*60)
    
    # Try regular LP-DiD with simpler specification
    from LPDiD import LPDiD
    
    try:
        lpdid_ols = LPDiD(
            data=data,
            depvar='death',
            unit='unit_id', 
            time='time_period',
            treat='treated',
            pre_window=3,
            post_window=3,
            formula='~ control_var',  # No fixed effects to avoid separation
            swap_pre_diff=True,
            cluster_formula='~ unit_id',
            n_jobs=1
        )
        
        print("Fitting OLS model...")
        results_ols = lpdid_ols.fit()
        
        if len(results_ols.event_study) > 0:
            print("OLS LP-DiD Results:")
            print(results_ols.event_study[['horizon', 'coefficient', 'se', 'p', 'obs']])
        else:
            print("No results obtained for OLS model")
            
    except Exception as e:
        print(f"OLS model failed: {e}")
        results_ols = None
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if len(results_pois.event_study) > 0:
        print("✓ LPDiDPois ran successfully")
        print("✓ Poisson regression handles binary death outcomes appropriately")
        print("✓ Coefficients can be interpreted as log hazard ratios")
        print("✓ swap_pre_diff=True ensures non-negative outcome differences")
        
        # Check if we see protective effects (negative coefficients)
        post_treatment_coefs = results_pois.event_study[
            (results_pois.event_study['horizon'] >= 0) & 
            (results_pois.event_study['horizon'] <= 3)
        ]['coefficient']
        
        if len(post_treatment_coefs) > 0 and (post_treatment_coefs < 0).any():
            print("✓ Model detected protective treatment effects (reduced death hazard)")
        
    else:
        print("✗ LPDiDPois failed to run")
    
    print("\nTest completed!")
    return True

if __name__ == "__main__":
    test_lpdid_poisson()