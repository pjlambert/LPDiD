"""
Example usage of the lpdid package with formula interface and wild bootstrap
"""

import numpy as np
import pandas as pd
from lpdid import LPDiD
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def generate_did_data(n_units=100, n_periods=20, treatment_period=10, 
                      effect_size=2.0, absorbing=True, n_clusters=None):
    """Generate simulated DiD data with optional clustering"""
    
    # Create panel structure
    units = np.repeat(range(n_units), n_periods)
    time = np.tile(range(n_periods), n_units)
    
    # Add clustering structure if requested
    if n_clusters:
        cluster1 = np.repeat(np.random.randint(0, n_clusters, n_units), n_periods)
        cluster2 = np.repeat(np.random.randint(0, n_clusters//2, n_units), n_periods)
    
    # Randomly assign treatment (50% treated)
    treated_units = np.random.choice(n_units, size=n_units//2, replace=False)
    
    # Create treatment indicator
    if absorbing:
        # Absorbing treatment
        treat = ((np.isin(units, treated_units)) & 
                (time >= treatment_period)).astype(int)
    else:
        # Non-absorbing treatment (units can switch in and out)
        treat = np.zeros(len(units))
        for u in treated_units:
            unit_mask = units == u
            # Random treatment periods
            treat_periods = np.random.choice(
                range(treatment_period, n_periods), 
                size=np.random.randint(1, 5), 
                replace=False
            )
            for t in treat_periods:
                treat[(unit_mask) & (time == t)] = 1
    
    # Generate outcome
    # Unit fixed effects
    unit_fe = np.random.normal(0, 1, n_units)
    # Time fixed effects  
    time_fe = np.random.normal(0, 0.5, n_periods) + 0.1 * np.arange(n_periods)
    
    # Industry fixed effect (categorical)
    industry = np.repeat(np.random.choice(['Tech', 'Finance', 'Retail'], n_units), n_periods)
    industry_fe = {'Tech': 0, 'Finance': 0.5, 'Retail': -0.5}
    industry_effect = np.array([industry_fe[ind] for ind in industry])
    
    # Control variables
    x1 = np.random.normal(0, 1, len(units))
    x2 = np.random.binomial(1, 0.5, len(units))
    
    # Base outcome
    y = (unit_fe[units] + time_fe[time] + industry_effect +
         0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.5, len(units)))
    
    # Add treatment effect (with dynamic pattern)
    for h in range(n_periods):
        future_treat = np.zeros(len(units))
        for t in range(n_periods - h):
            if t + h < n_periods:
                idx = time == t
                future_idx = time == t + h
                future_treat[idx] = treat[future_idx]
        
        # Dynamic treatment effect
        if h == 0:
            effect = 0  # No immediate effect
        elif h <= 3:
            effect = effect_size * h / 3  # Gradual increase
        else:
            effect = effect_size  # Stable effect
        
        # Apply effect to switchers
        switchers = (treat == 0) & (future_treat == 1)
        y[switchers] += effect
    
    # Create weights (e.g., based on unit size)
    weights = np.repeat(np.random.uniform(0.5, 2.0, n_units), n_periods)
    
    # Create DataFrame
    df = pd.DataFrame({
        'unit': units,
        'time': time,
        'treat': treat,
        'y': y,
        'x1': x1,
        'x2': x2,
        'industry': industry,
        'weights': weights
    })
    
    if n_clusters:
        df['cluster1'] = cluster1
        df['cluster2'] = cluster2
    
    return df

# Example 1: Basic formula interface
print("=" * 60)
print("Example 1: Formula Interface")
print("=" * 60)

# Generate data
df = generate_did_data(n_units=200, n_periods=30, treatment_period=15, 
                       effect_size=3.0, n_clusters=20)

# Estimate LP-DiD with formula interface
lpdid = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",  # Controls and fixed effects
    n_jobs=-1
)

results = lpdid.fit()

print("\nEvent Study Results:")
print(results.event_study)

print("\nPooled Results:")
print(results.pooled)

# Plot
fig, ax = results.plot(title='LP-DiD with Formula Interface')
plt.show()

# Example 2: Wild Bootstrap with Multi-way Clustering
print("\n" + "=" * 60)
print("Example 2: Wild Bootstrap with Multi-way Clustering")
print("=" * 60)

lpdid_wb = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    cluster_formula="~ cluster1 + cluster2",  # Multi-way clustering
    wildbootstrap=999,  # Wild bootstrap iterations
    seed=123,
    n_jobs=-1
)

results_wb = lpdid_wb.fit()

print("\nWild Bootstrap Results:")
print(results_wb.event_study[['horizon', 'coefficient', 'se', 'p', 'ci_low', 'ci_high']])

# Compare standard errors
print("\nStandard Error Comparison:")
print("Analytical SE (horizon 3):", results.event_study[results.event_study['horizon']==3]['se'].values[0])
print("Wild Bootstrap SE (horizon 3):", results_wb.event_study[results_wb.event_study['horizon']==3]['se'].values[0])

# Example 3: Weighted Estimation with Formula
print("\n" + "=" * 60)
print("Example 3: Weighted Estimation")
print("=" * 60)

lpdid_weighted = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    weights='weights',  # Use weights variable
    n_jobs=-1
)

results_weighted = lpdid_weighted.fit()

print("\nWeighted Estimation Results:")
print(results_weighted.pooled)

# Example 4: Complex Formula with Interactions
print("\n" + "=" * 60)
print("Example 4: Complex Formulas")
print("=" * 60)

# Add interaction term
df['x1_x2'] = df['x1'] * df['x2']

lpdid_complex = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 + x1_x2 | industry",  # Include interaction
    ylags=2,  # Add outcome lags
    wildbootstrap=499,
    n_jobs=-1
)

results_complex = lpdid_complex.fit()

print("\nComplex Model Results (selected horizons):")
selected = results_complex.event_study[results_complex.event_study['horizon'].isin([-3, 0, 3, 5])]
print(selected[['horizon', 'coefficient', 'se', 'p']])

# Example 5: Reweighted ATE with Wild Bootstrap
print("\n" + "=" * 60)
print("Example 5: Reweighted ATE with Wild Bootstrap")
print("=" * 60)

lpdid_rw_wb = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    rw=True,  # Reweight for equally-weighted ATE
    wildbootstrap=999,
    weights='weights',  # Combine with user weights
    n_jobs=-1
)

results_rw_wb = lpdid_rw_wb.fit()

print("\nReweighted + Wild Bootstrap Results:")
print(results_rw_wb.pooled)

# Example 6: PMD with Formula Interface
print("\n" + "=" * 60)
print("Example 6: Pre-mean Differencing with Formulas")
print("=" * 60)

lpdid_pmd = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    pmd='max',  # Use all pre-treatment periods
    cluster_formula="~ cluster1",  # Single clustering
    wildbootstrap=499,
    n_jobs=-1
)

results_pmd = lpdid_pmd.fit()

print("\nPMD Results (horizon 5):")
h5 = results_pmd.event_study[results_pmd.event_study['horizon'] == 5]
print(f"Coefficient: {h5['coefficient'].values[0]:.3f}")
print(f"Wild Bootstrap SE: {h5['se'].values[0]:.3f}")
print(f"95% CI: [{h5['ci_low'].values[0]:.3f}, {h5['ci_high'].values[0]:.3f}]")

# Example 7: Non-absorbing Treatment with Formula
print("\n" + "=" * 60)
print("Example 7: Non-absorbing Treatment with Formulas")
print("=" * 60)

# Generate non-absorbing data
df_nonabs = generate_did_data(n_units=200, n_periods=30, treatment_period=10,
                              effect_size=2.0, absorbing=False, n_clusters=15)

lpdid_nonabs = LPDiD(
    data=df_nonabs,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ x1 + x2 | industry",
    nonabsorbing=(5, False, False),  # L=5
    cluster_formula="~ cluster1",
    wildbootstrap=499,
    n_jobs=-1
)

results_nonabs = lpdid_nonabs.fit()

print("\nNon-absorbing Treatment Results:")
print(results_nonabs.event_study[['horizon', 'coefficient', 'se', 'p']].head(10))

# Example 8: Comparison of different specifications
print("\n" + "=" * 60)
print("Example 8: Specification Comparison")
print("=" * 60)

specs = {
    'Base': {},
    'Controls': {'formula': '~ x1 + x2'},
    'FE': {'formula': '~ x1 + x2 | industry'},
    'Wild Bootstrap': {'formula': '~ x1 + x2 | industry', 'wildbootstrap': 999},
    'Multi-cluster': {'formula': '~ x1 + x2 | industry', 
                     'cluster_formula': '~ cluster1 + cluster2',
                     'wildbootstrap': 999},
    'Weighted': {'formula': '~ x1 + x2 | industry', 'weights': 'weights'},
    'Reweighted': {'formula': '~ x1 + x2 | industry', 'rw': True}
}

comparison_results = []

for name, spec_kwargs in specs.items():
    print(f"\nRunning: {name}")
    
    lpdid_spec = LPDiD(
        data=df,
        depvar='y',
        unit='unit',
        time='time',
        treat='treat',
        pre_window=5,
        post_window=10,
        n_jobs=-1,
        **spec_kwargs
    )
    
    res = lpdid_spec.fit()
    
    # Extract horizon 5 results
    h5 = res.event_study[res.event_study['horizon'] == 5].iloc[0]
    
    comparison_results.append({
        'Specification': name,
        'Coefficient': h5['coefficient'],
        'SE': h5['se'],
        'CI_Low': h5['ci_low'],
        'CI_High': h5['ci_high']
    })

comparison_df = pd.DataFrame(comparison_results)
print("\n" + "=" * 60)
print("Specification Comparison (Horizon 5):")
print("=" * 60)
print(comparison_df.to_string(index=False, float_format='%.3f'))

def generate_did_data(n_units=100, n_periods=20, treatment_period=10, 
                      effect_size=2.0, absorbing=True):
    """Generate simulated DiD data"""
    
    # Create panel structure
    units = np.repeat(range(n_units), n_periods)
    time = np.tile(range(n_periods), n_units)
    
    # Randomly assign treatment (50% treated)
    treated_units = np.random.choice(n_units, size=n_units//2, replace=False)
    
    # Create treatment indicator
    if absorbing:
        # Absorbing treatment
        treat = ((np.isin(units, treated_units)) & 
                (time >= treatment_period)).astype(int)
    else:
        # Non-absorbing treatment (units can switch in and out)
        treat = np.zeros(len(units))
        for u in treated_units:
            unit_mask = units == u
            # Random treatment periods
            treat_periods = np.random.choice(
                range(treatment_period, n_periods), 
                size=np.random.randint(1, 5), 
                replace=False
            )
            for t in treat_periods:
                treat[(unit_mask) & (time == t)] = 1
    
    # Generate outcome
    # Unit fixed effects
    unit_fe = np.random.normal(0, 1, n_units)
    # Time fixed effects  
    time_fe = np.random.normal(0, 0.5, n_periods) + 0.1 * np.arange(n_periods)
    
    # Base outcome
    y = unit_fe[units] + time_fe[time] + np.random.normal(0, 0.5, len(units))
    
    # Add treatment effect (with dynamic pattern)
    for h in range(n_periods):
        future_treat = np.zeros(len(units))
        for t in range(n_periods - h):
            if t + h < n_periods:
                idx = time == t
                future_idx = time == t + h
                future_treat[idx] = treat[future_idx]
        
        # Dynamic treatment effect
        if h == 0:
            effect = 0  # No immediate effect
        elif h <= 3:
            effect = effect_size * h / 3  # Gradual increase
        else:
            effect = effect_size  # Stable effect
        
        # Apply effect to switchers
        switchers = (treat == 0) & (future_treat == 1)
        y[switchers] += effect
    
    # Create DataFrame
    df = pd.DataFrame({
        'unit': units,
        'time': time,
        'treat': treat,
        'y': y
    })
    
    return df

# Example 1: Absorbing treatment
print("=" * 60)
print("Example 1: Absorbing Treatment")
print("=" * 60)

# Generate data
df_absorbing = generate_did_data(
    n_units=200, 
    n_periods=30, 
    treatment_period=15,
    effect_size=3.0,
    absorbing=True
)

# Estimate LP-DiD
lpdid_absorbing = LPDiD(
    data=df_absorbing,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    n_jobs=-1  # Use all available cores
)

# Fit the model
results_absorbing = lpdid_absorbing.fit()

# Display results
print("\nEvent Study Results:")
print(results_absorbing.event_study)

print("\nPooled Results:")
print(results_absorbing.pooled)

# Plot
fig, ax = results_absorbing.plot(title='LP-DiD Event Study: Absorbing Treatment')
plt.show()

# Example 2: Reweighted estimation with controls
print("\n" + "=" * 60)
print("Example 2: Reweighted LP-DiD with Controls")
print("=" * 60)

# Add some covariates
df_absorbing['x1'] = np.random.normal(0, 1, len(df_absorbing))
df_absorbing['x2'] = np.random.binomial(1, 0.5, len(df_absorbing))

# Estimate with reweighting and controls
lpdid_rw = LPDiD(
    data=df_absorbing,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    controls=['x1', 'x2'],
    rw=True,  # Reweight for equally-weighted ATE
    nocomp=True,  # Avoid composition changes
    n_jobs=-1
)

results_rw = lpdid_rw.fit()

print("\nReweighted Event Study Results:")
print(results_rw.event_study)

# Example 3: Non-absorbing treatment
print("\n" + "=" * 60)
print("Example 3: Non-absorbing Treatment")
print("=" * 60)

# Generate non-absorbing data
df_nonabsorbing = generate_did_data(
    n_units=200,
    n_periods=30,
    treatment_period=10,
    effect_size=2.0,
    absorbing=False
)

# Estimate with non-absorbing treatment
lpdid_nonabsorbing = LPDiD(
    data=df_nonabsorbing,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    nonabsorbing=(5, False, False),  # L=5, notyet=False, firsttreat=False
    n_jobs=-1
)

results_nonabsorbing = lpdid_nonabsorbing.fit()

print("\nNon-absorbing Treatment Results:")
print(results_nonabsorbing.event_study)

# Example 4: Using never-treated as controls
print("\n" + "=" * 60)
print("Example 4: Never-treated Control Group")
print("=" * 60)

lpdid_never = LPDiD(
    data=df_absorbing,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    nevertreated=True,
    n_jobs=-1
)

results_never = lpdid_never.fit(only_event=True)  # Only event study

print("\nUsing never-treated units as controls:")
print(results_never.event_study)

# Example 5: Pre-mean differencing (PMD)
print("\n" + "=" * 60)
print("Example 5: Pre-mean Differencing (PMD)")
print("=" * 60)

lpdid_pmd = LPDiD(
    data=df_absorbing,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    pmd='max',  # Use all pre-treatment periods
    n_jobs=-1
)

results_pmd = lpdid_pmd.fit()

print("\nPMD Results:")
print(results_pmd.event_study)

# Example 6: Custom pooled windows
print("\n" + "=" * 60)
print("Example 6: Custom Pooled Windows")
print("=" * 60)

results_custom = lpdid_absorbing.fit(
    post_pooled=(0, 5),  # Average effect for periods 0-5
    pre_pooled=(2, 4)    # Pre-trend test for periods -4 to -2
)

print("\nCustom Pooled Results:")
print(results_custom.pooled)

# Example 7: Parallel processing comparison
print("\n" + "=" * 60)
print("Example 7: Parallel Processing Speed Test")
print("=" * 60)

import time

# Sequential processing
lpdid_seq = LPDiD(
    data=df_absorbing,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=10,
    post_window=20,
    n_jobs=1  # Sequential
)

start_time = time.time()
results_seq = lpdid_seq.fit()
seq_time = time.time() - start_time

print(f"Sequential processing time: {seq_time:.2f} seconds")

# Parallel processing
lpdid_par = LPDiD(
    data=df_absorbing,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=10,
    post_window=20,
    n_jobs=-1  # All cores
)

start_time = time.time()
results_par = lpdid_par.fit()
par_time = time.time() - start_time

print(f"Parallel processing time: {par_time:.2f} seconds")
print(f"Speed-up: {seq_time/par_time:.2f}x")

# Verify results are identical
print(f"\nResults match: {np.allclose(results_seq.event_study['coefficient'], results_par.event_study['coefficient'])}")