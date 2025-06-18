import numpy as np
import pandas as pd
import os
from scipy import integrate

def generate_duration_data(n=1000, T=20, t_star=11, initial_mean_g1=0.4, 
                          initial_mean_g2=0.2, c=0.5, beta=1.0, p=None, seed=None):
    """
    Generate individual-level panel data with duration/absorbing state structure
    following the data generating process from Deaner & Ku (2024).
    
    Parameters:
    -----------
    n : int
        Number of individuals
    T : int
        Number of time periods
    t_star : int
        Treatment time (group 1 gets treated at this time)
    initial_mean_g1 : float
        Target E[Y_{i,1}|G_i = 1] - expected outcome at t=1 for group 1
    initial_mean_g2 : float
        Target E[Y_{i,1}|G_i = 2] - expected outcome at t=1 for group 2
    c : float
        Level difference in hazard rates between groups
    beta : float
        Treatment effect on hazard rate
    p : float or None
        Parameter for hazard function shape. If None, will be calibrated.
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame with columns:
        - i: individual identifier (0 to n-1)
        - t: time period (1 to T)
        - group: group membership (1 or 2)
        - Y: observed outcome (0 or 1) - this is the main outcome variable
        - Y_counterfactual: counterfactual outcome under no treatment (0 or 1)
        - treated: indicator for whether individual i is treated at time t
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Hazard function as specified in the paper
    def hazard_function(t, group, treatment=False):
        base_hazard = 1 + p * t / T - 0.5 * (t / T - 0.5)**2
        
        if group == 1:
            base_hazard += c
            if treatment and t >= t_star:
                base_hazard += beta
        
        return base_hazard / (T - 1)
    
    # Compute transition probability between two time points
    def transition_probability(t1, t2, group, treatment=False):
        """P(Y_{t2} = 1 | Y_{t1} = 0, G = group)"""
        cumulative_hazard, _ = integrate.quad(
            lambda s: hazard_function(s, group, treatment), t1, t2
        )
        return 1 - np.exp(-cumulative_hazard)
    
    # Calibrate p if not provided
    if p is None:
        # Simple grid search to match initial means
        p_values = np.linspace(-2, 2, 100)
        best_p = None
        best_error = float('inf')
        
        for p_candidate in p_values:
            p = p_candidate
            
            # Compute P(Y_1 = 1|G = k) for each group
            prob_g1 = transition_probability(0, 1, group=1, treatment=False)
            prob_g2 = transition_probability(0, 1, group=2, treatment=False)
            
            error = (prob_g1 - initial_mean_g1)**2 + (prob_g2 - initial_mean_g2)**2
            
            if error < best_error:
                best_error = error
                best_p = p_candidate
        
        p = best_p
    
    # Assign individuals to groups (50-50 split)
    groups = np.random.choice([1, 2], size=n, p=[0.5, 0.5])
    
    # Generate individual-level panel data
    data = []
    
    for i in range(n):
        group = groups[i]
        
        # Track states (start with Y=0 for everyone)
        y_observed = 0
        y_counterfactual = 0
        
        for t in range(1, T + 1):
            # Determine treatment status
            treated = (group == 1) and (t >= t_star)
            
            # Generate observed outcome
            if y_observed == 0:  # Can only transition if not already in state 1
                prob_transition = transition_probability(t-1, t, group, treatment=(group==1 and treated))
                if np.random.random() < prob_transition:
                    y_observed = 1
            
            # Generate counterfactual outcome (no treatment for anyone)
            if y_counterfactual == 0:
                prob_transition_cf = transition_probability(t-1, t, group, treatment=False)
                if np.random.random() < prob_transition_cf:
                    y_counterfactual = 1
            
            # Store observation
            data.append({
                'i': i,
                't': t,
                'group': group,
                'Y': y_observed,
                'Y_counterfactual': y_counterfactual,
                'treated': int(treated)
            })
    
    df = pd.DataFrame(data)
    
    # Print some basic statistics to verify the data
    print("Data generation complete!")
    print(f"Number of individuals: {n}")
    print(f"Number of time periods: {T}")
    print(f"Treatment begins at t={t_star}")
    print(f"\nGroup sizes:")
    print(df[df['t']==1].groupby('group').size())
    
    # Check initial period outcomes match targets
    initial_outcomes = df[df['t']==1].groupby('group')['Y'].mean()
    print(f"\nInitial outcomes (t=1):")
    print(f"Group 1: {initial_outcomes[1]:.3f} (target: {initial_mean_g1})")
    print(f"Group 2: {initial_outcomes[2]:.3f} (target: {initial_mean_g2})")
    
    # Check absorbing state property
    violations = 0
    for i in range(n):
        individual_data = df[df['i'] == i].sort_values('t')
        y_values = individual_data['Y'].values
        # Check if Y ever goes from 1 to 0
        for j in range(1, len(y_values)):
            if y_values[j-1] == 1 and y_values[j] == 0:
                violations += 1
    print(f"\nAbsorbing state violations: {violations} (should be 0)")
    
    return df

def generate_and_save_dataset(params, filename, output_dir):
    """
    Generate a dataset with the given parameters and save it to the specified file.
    
    Parameters:
    -----------
    params : dict
        Dictionary of parameters to pass to generate_duration_data
    filename : str
        Name of the file to save to (without directory)
    output_dir : str
        Directory to save the file in
    """
    print(f"\nGenerating dataset: {filename}")
    print(f"Parameters: {params}")
    
    df = generate_duration_data(**params)
    
    # Calculate and print some statistics
    final_period = df[df['t'] == params.get('T', 20)]
    treated_group = final_period[final_period['group'] == 1]
    ate = (treated_group['Y'] - treated_group['Y_counterfactual']).mean()
    print(f"Average treatment effect at t={params.get('T', 20)}: {ate:.3f}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")
    
    return df

# Main execution block
if __name__ == "__main__":
    # Define the output directory
    output_dir = os.path.join(os.path.dirname(__file__), "synth_data")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating synthetic datasets for LPDiDPois testing")
    print(f"Output directory: {output_dir}")
    
    # Define several parameter sets for different datasets with more dramatic treatment effects
    dataset_params = [
        # Dataset 1: Baseline parameters with stronger treatment effect
        {
            "name": "baseline",
            "params": {
                "n": 5000,
                "T": 20,
                "t_star": 11,
                "initial_mean_g1": 0.4,
                "initial_mean_g2": 0.2,
                "c": 0.8,       # Increased from 0.5
                "beta": 3.0,    # Increased from 1.0 to 3.0
                "seed": 42
            }
        },
        # Dataset 2: Very large treatment effect
        {
            "name": "high_treatment_effect",
            "params": {
                "n": 5000,
                "T": 20,
                "t_star": 11,
                "initial_mean_g1": 0.4,
                "initial_mean_g2": 0.2,
                "c": 0.8,
                "beta": 5.0,    # Increased from 2.0 to 5.0
                "seed": 43
            }
        },
        # Dataset 3: Earlier treatment time with strong effect
        {
            "name": "early_treatment",
            "params": {
                "n": 5000,
                "T": 20,
                "t_star": 6,
                "initial_mean_g1": 0.4,
                "initial_mean_g2": 0.2,
                "c": 0.8,
                "beta": 3.0,    # Increased from 1.0 to 3.0
                "seed": 44
            }
        },
        # Dataset 4: More similar groups at baseline but strong treatment effect
        {
            "name": "similar_groups",
            "params": {
                "n": 5000,
                "T": 20,
                "t_star": 11,
                "initial_mean_g1": 0.3,
                "initial_mean_g2": 0.25,
                "c": 0.2,
                "beta": 3.5,    # Increased from 1.0 to 3.5
                "seed": 45
            }
        },
        # Dataset 5: Smaller sample size with strong effect
        {
            "name": "small_sample",
            "params": {
                "n": 1000,
                "T": 20,
                "t_star": 11,
                "initial_mean_g1": 0.4,
                "initial_mean_g2": 0.2,
                "c": 0.8,
                "beta": 3.0,    # Increased from 1.0 to 3.0
                "seed": 46
            }
        },
        # Dataset 6: Longer time horizon with strong effect
        {
            "name": "long_horizon",
            "params": {
                "n": 5000,
                "T": 30,
                "t_star": 16,
                "initial_mean_g1": 0.4,
                "initial_mean_g2": 0.2,
                "c": 0.8,
                "beta": 3.0,    # Increased from 1.0 to 3.0
                "seed": 47
            }
        },
        # Dataset 7: Strong negative treatment effect
        {
            "name": "negative_effect",
            "params": {
                "n": 5000,
                "T": 20,
                "t_star": 11,
                "initial_mean_g1": 0.4,
                "initial_mean_g2": 0.2,
                "c": 0.8,
                "beta": -2.5,   # Changed from -0.5 to -2.5 (more dramatic)
                "seed": 48
            }
        }
    ]
    
    # Generate and save each dataset
    for dataset in dataset_params:
        filename = f"synth_data_{dataset['name']}.csv"
        generate_and_save_dataset(dataset['params'], filename, output_dir)
    
    print("\nAll synthetic datasets generated successfully!")