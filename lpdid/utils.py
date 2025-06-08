"""
Utility functions for lpdid package
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def generate_panel_data(
    n_units: int = 100,
    n_periods: int = 20,
    treatment_start: int = 10,
    treatment_effect: float = 1.0,
    unit_fe_sd: float = 1.0,
    time_fe_sd: float = 0.5,
    idiosyncratic_sd: float = 1.0,
    treatment_share: float = 0.5,
    absorbing: bool = True,
    dynamic_effects: Optional[dict] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate simulated panel data for testing LP-DiD
    
    Parameters
    ----------
    n_units : int
        Number of units
    n_periods : int
        Number of time periods
    treatment_start : int
        Period when treatment starts
    treatment_effect : float
        Size of treatment effect
    unit_fe_sd : float
        Standard deviation of unit fixed effects
    time_fe_sd : float
        Standard deviation of time fixed effects
    idiosyncratic_sd : float
        Standard deviation of idiosyncratic errors
    treatment_share : float
        Share of units that get treated (0 to 1)
    absorbing : bool
        Whether treatment is absorbing
    dynamic_effects : dict, optional
        Dictionary mapping relative time to effect size
    seed : int, optional
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Panel data with columns: unit, time, treat, y
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create panel structure
    units = np.repeat(range(n_units), n_periods)
    time = np.tile(range(n_periods), n_units)
    
    # Randomly assign treatment status
    n_treated = int(n_units * treatment_share)
    treated_units = np.random.choice(n_units, size=n_treated, replace=False)
    
    # Create treatment indicator
    if absorbing:
        treat = ((np.isin(units, treated_units)) & 
                (time >= treatment_start)).astype(int)
    else:
        # Non-absorbing: units can switch in and out
        treat = np.zeros(len(units))
        for u in treated_units:
            unit_mask = units == u
            # Each treated unit has 2-5 treatment episodes
            n_episodes = np.random.randint(2, 6)
            treat_periods = np.random.choice(
                range(treatment_start, n_periods),
                size=min(n_episodes, n_periods - treatment_start),
                replace=False
            )
            for t in treat_periods:
                treat[(unit_mask) & (time == t)] = 1
    
    # Generate fixed effects
    unit_fe = np.random.normal(0, unit_fe_sd, n_units)
    time_fe = np.random.normal(0, time_fe_sd, n_periods)
    
    # Add time trend
    time_trend = 0.1 * np.arange(n_periods)
    
    # Generate base outcome
    y = (unit_fe[units] + 
         time_fe[time] + 
         time_trend[time] +
         np.random.normal(0, idiosyncratic_sd, len(units)))
    
    # Add treatment effects
    if dynamic_effects is None:
        # Default dynamic pattern
        dynamic_effects = {
            -2: 0,    # No pre-trend
            -1: 0,    # No pre-trend
            0: 0.3 * treatment_effect,   # Partial immediate effect
            1: 0.7 * treatment_effect,   # Building up
            2: 0.9 * treatment_effect,   # Almost full
            3: treatment_effect,         # Full effect
            4: treatment_effect,         # Persistent
            5: treatment_effect,         # Persistent
        }
    
    # Apply dynamic effects
    df_temp = pd.DataFrame({
        'unit': units,
        'time': time,
        'treat': treat,
        'y_base': y
    })
    
    # Calculate treatment effects for each observation
    for idx, row in df_temp.iterrows():
        if row['treat'] == 0:
            continue
            
        # Find when unit first got treated
        unit_treat_history = df_temp[
            (df_temp['unit'] == row['unit']) & 
            (df_temp['treat'] == 1)
        ]['time'].values
        
        if len(unit_treat_history) > 0:
            first_treat_time = unit_treat_history.min()
            relative_time = row['time'] - first_treat_time
            
            # Apply dynamic effect if specified
            if relative_time in dynamic_effects:
                y[idx] += dynamic_effects[relative_time]
            elif relative_time > max(dynamic_effects.keys()):
                # Use the last specified effect for all future periods
                y[idx] += dynamic_effects[max(dynamic_effects.keys())]
    
    # Create final DataFrame
    df = pd.DataFrame({
        'unit': units,
        'time': time,
        'treat': treat,
        'y': y
    })
    
    return df


def simulate_lpdid_power(
    n_simulations: int = 100,
    n_units: int = 100,
    n_periods: int = 20,
    treatment_start: int = 10,
    treatment_effect: float = 1.0,
    pre_window: int = 5,
    post_window: int = 5,
    **kwargs
) -> pd.DataFrame:
    """
    Run power analysis for LP-DiD
    
    Parameters
    ----------
    n_simulations : int
        Number of simulations
    n_units : int
        Number of units per simulation
    n_periods : int
        Number of periods
    treatment_start : int
        When treatment starts
    treatment_effect : float
        True treatment effect
    pre_window : int
        Pre-treatment window for estimation
    post_window : int  
        Post-treatment window for estimation
    **kwargs
        Additional arguments passed to generate_panel_data
        
    Returns
    -------
    pd.DataFrame
        Results from all simulations
    """
    from lpdid import LPDiD
    
    results = []
    
    for sim in range(n_simulations):
        # Generate data
        df = generate_panel_data(
            n_units=n_units,
            n_periods=n_periods,
            treatment_start=treatment_start,
            treatment_effect=treatment_effect,
            seed=sim,
            **kwargs
        )
        
        # Estimate
        try:
            lpdid = LPDiD(
                data=df,
                depvar='y',
                unit='unit',
                time='time',
                treat='treat',
                pre_window=pre_window,
                post_window=post_window
            )
            
            res = lpdid.fit()
            
            # Extract key results
            for _, row in res.event_study.iterrows():
                results.append({
                    'simulation': sim,
                    'horizon': row['horizon'],
                    'coefficient': row['coefficient'],
                    'se': row['se'],
                    'p': row['p'],
                    'ci_low': row['ci_low'],
                    'ci_high': row['ci_high'],
                    'true_effect': treatment_effect if row['horizon'] >= 3 else 0
                })
                
        except Exception as e:
            print(f"Simulation {sim} failed: {e}")
            continue
    
    return pd.DataFrame(results)


def plot_power_analysis(power_results: pd.DataFrame, true_effect: float = 1.0):
    """
    Plot results from power analysis
    
    Parameters
    ----------
    power_results : pd.DataFrame
        Results from simulate_lpdid_power
    true_effect : float
        True treatment effect for comparison
    """
    import matplotlib.pyplot as plt
    
    # Calculate coverage and power by horizon
    summary = []
    for h in power_results['horizon'].unique():
        h_data = power_results[power_results['horizon'] == h]
        
        # Coverage: does CI contain true effect?
        if h >= 0:
            true_val = true_effect if h >= 3 else true_effect * h / 3
        else:
            true_val = 0
            
        coverage = ((h_data['ci_low'] <= true_val) & 
                   (h_data['ci_high'] >= true_val)).mean()
        
        # Power: reject null when there is an effect
        if true_val != 0:
            power = (h_data['p'] < 0.05).mean()
        else:
            power = np.nan
            
        # Size: reject null when there is no effect  
        if true_val == 0:
            size = (h_data['p'] < 0.05).mean()
        else:
            size = np.nan
            
        summary.append({
            'horizon': h,
            'mean_coef': h_data['coefficient'].mean(),
            'sd_coef': h_data['coefficient'].std(),
            'coverage': coverage,
            'power': power,
            'size': size,
            'true_effect': true_val
        })
    
    summary_df = pd.DataFrame(summary).sort_values('horizon')
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Average coefficients
    ax = axes[0, 0]
    ax.plot(summary_df['horizon'], summary_df['mean_coef'], 
            'o-', label='Average estimate', color='navy')
    ax.plot(summary_df['horizon'], summary_df['true_effect'], 
            '--', label='True effect', color='red')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=-0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Coefficient')
    ax.set_title('Average Treatment Effect Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Coverage
    ax = axes[0, 1]
    ax.plot(summary_df['horizon'], summary_df['coverage'], 
            'o-', color='green')
    ax.axhline(y=0.95, color='red', linestyle='--', 
               label='Nominal 95%')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Coverage')
    ax.set_title('Confidence Interval Coverage')
    ax.set_ylim(0.8, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Power/Size
    ax = axes[1, 0]
    post_data = summary_df[summary_df['horizon'] >= 0]
    pre_data = summary_df[summary_df['horizon'] < 0]
    
    if not post_data.empty:
        ax.plot(post_data['horizon'], post_data['power'], 
                'o-', label='Power (post)', color='blue')
    if not pre_data.empty:
        ax.plot(pre_data['horizon'], pre_data['size'], 
                's-', label='Size (pre)', color='orange')
    
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Rejection rate')
    ax.set_title('Power and Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Distribution at specific horizon
    ax = axes[1, 1]
    h3_data = power_results[power_results['horizon'] == 3]['coefficient']
    if len(h3_data) > 0:
        ax.hist(h3_data, bins=30, density=True, alpha=0.7, color='navy')
        ax.axvline(x=true_effect, color='red', linestyle='--', 
                   label=f'True effect = {true_effect}')
        ax.set_xlabel('Coefficient')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Estimates at Horizon 3')
        ax.legend()
    
    plt.tight_layout()
    return fig, summary_df


# Convenience function for quick testing
def quick_test_lpdid():
    """Quick test of lpdid with simulated data"""
    from lpdid import LPDiD
    
    # Generate data
    print("Generating test data...")
    df = generate_panel_data(
        n_units=100,
        n_periods=30,
        treatment_start=15,
        treatment_effect=2.0,
        seed=42
    )
    
    print(f"Data shape: {df.shape}")
    print(f"Treatment share: {df['treat'].mean():.2%}")
    
    # Estimate
    print("\nEstimating LP-DiD...")
    lpdid = LPDiD(
        data=df,
        depvar='y',
        unit='unit',
        time='time',
        treat='treat',
        pre_window=5,
        post_window=10,
        n_jobs=-1
    )
    
    results = lpdid.fit()
    
    # Display results
    print("\nEvent Study Results:")
    print(results.event_study)
    
    print("\nPooled Results:")
    print(results.pooled)
    
    # Plot
    print("\nCreating plot...")
    fig, ax = results.plot(title='LP-DiD Test Results')
    
    return results, fig