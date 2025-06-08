"""
Example usage of LP-DiD with IV and interaction features
"""

import numpy as np
import pandas as pd
from lpdid import LPDiD
import warnings

# Set random seed for reproducibility
np.random.seed(42)

def generate_iv_did_data(n_units=200, n_periods=30, treatment_period=15):
    """Generate DiD data with endogenous treatment and instruments"""
    
    # Create panel structure
    units = np.repeat(range(n_units), n_periods)
    time = np.tile(range(n_periods), n_units)
    
    # Unit characteristics
    unit_quality = np.repeat(np.random.normal(0, 1, n_units), n_periods)
    unit_size = np.repeat(np.random.uniform(0.5, 2.0, n_units), n_periods)
    
    # Industry categories
    industries = np.repeat(np.random.choice(['Tech', 'Finance', 'Retail'], n_units), n_periods)
    industry_dummies = pd.get_dummies(industries, prefix='ind')
    
    # Create instruments
    # Instrument 1: Eligibility threshold based on size
    threshold = 1.2
    eligible = (unit_size > threshold).astype(int)
    
    # Instrument 2: Geographic spillover (neighboring units' treatment)
    np.random.seed(42)
    neighbor_treat_intensity = np.random.uniform(0, 1, len(units))
    
    # Unobserved confounders that affect both treatment and outcome
    unobserved = np.repeat(np.random.normal(0, 1, n_units), n_periods)
    
    # Treatment is endogenous - affected by unobservables
    # But also affected by instruments
    treat_propensity = (
        0.5 * eligible + 
        0.3 * neighbor_treat_intensity + 
        0.4 * unobserved +  # This creates endogeneity
        0.2 * unit_quality +
        np.random.normal(0, 0.5, len(units))
    )
    
    # Treatment indicator (absorbing)
    treat = np.zeros(len(units))
    for i in range(n_units):
        unit_mask = units == i
        unit_propensity = treat_propensity[unit_mask]
        # Unit gets treated when propensity exceeds threshold after treatment period
        treat_time = np.where((time[unit_mask] >= treatment_period) & 
                              (unit_propensity > 0.5))[0]
        if len(treat_time) > 0:
            first_treat = treat_time[0] + treatment_period
            treat[(unit_mask) & (time >= first_treat)] = 1
    
    # Generate outcome
    # True treatment effect varies by size (heterogeneity)
    true_effect_by_size = 2.0 + 1.5 * unit_size
    
    # Industry fixed effects
    industry_fe = {'Tech': 1.0, 'Finance': 0.5, 'Retail': -0.5}
    ind_effect = np.array([industry_fe[ind] for ind in industries])
    
    # Time fixed effects
    time_fe = 0.1 * time + 0.01 * (time ** 2)
    
    # Outcome equation
    y = (
        2.0 +  # Constant
        0.5 * unit_quality +
        0.3 * unit_size +
        ind_effect +
        time_fe +
        0.8 * unobserved +  # Unobserved affects outcome too (creates bias)
        true_effect_by_size * treat +  # Heterogeneous treatment effect
        np.random.normal(0, 0.5, len(units))
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'unit': units,
        'time': time,
        'treat': treat,
        'y': y,
        'size': unit_size,
        'quality': unit_quality,
        'industry': industries,
        'eligible': eligible,
        'spillover': neighbor_treat_intensity,
        'unobserved': unobserved  # In practice, we wouldn't observe this
    })
    
    # Add industry dummies
    for col in industry_dummies.columns:
        df[col] = industry_dummies[col].values
    
    return df

# Generate data
print("Generating data with endogenous treatment...")
df = generate_iv_did_data(n_units=300, n_periods=30, treatment_period=15)

print(f"\nData shape: {df.shape}")
print(f"Treatment rate: {df['treat'].mean():.2%}")
print(f"Eligible rate: {df['eligible'].mean():.2%}")

# Example 1: OLS (biased due to endogeneity)
print("\n" + "="*70)
print("Example 1: Standard LP-DiD (OLS - Biased)")
print("="*70)

lpdid_ols = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ quality + size | industry",
    n_jobs=-1
)

results_ols = lpdid_ols.fit(only_pooled=True)
print("\nOLS Results (biased due to unobserved confounders):")
print(results_ols.pooled)

# Example 2: IV estimation
print("\n" + "="*70)
print("Example 2: LP-DiD with Instrumental Variables")
print("="*70)

lpdid_iv = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ quality + size | industry | D_treat ~ eligible + spillover",
    wildbootstrap=999,
    seed=123,
    n_jobs=-1
)

results_iv = lpdid_iv.fit()

print("\nIV Event Study Results (selected horizons):")
selected = results_iv.event_study[results_iv.event_study['horizon'].isin([-3, 0, 3, 5, 10])]
print(selected[['horizon', 'coefficient', 'se', 'p']])

print("\nIV Pooled Results:")
print(results_iv.pooled)

print("\nIV Diagnostics:")
if results_iv.iv_diagnostics is not None:
    print(results_iv.iv_diagnostics[results_iv.iv_diagnostics['horizon'].isin([0, 5, 10])])

print("\nFirst Stage Results (horizon 5):")
if results_iv.first_stage is not None:
    fs_h5 = results_iv.first_stage[results_iv.first_stage['horizon'] == 5]
    if not fs_h5.empty:
        print(fs_h5)

# Example 3: Treatment effect heterogeneity
print("\n" + "="*70)
print("Example 3: Treatment Effect Heterogeneity (Interactions)")
print("="*70)

lpdid_het = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ quality | industry",
    interactions="~ size",  # Interact treatment with size
    n_jobs=-1
)

results_het = lpdid_het.fit()

print("\nHeterogeneous Effects by Size (selected horizons):")
het_cols = ['horizon', 'coefficient', 'se', 'size_interaction', 'size_interaction_se', 'size_interaction_p']
selected_het = results_het.event_study[results_het.event_study['horizon'].isin([0, 3, 5, 10])]
print(selected_het[het_cols])

print("\nInterpretation:")
print("- 'coefficient': Effect when size = 0 (extrapolation)")
print("- 'size_interaction': How effect changes with unit size")
print("- Total effect = coefficient + size_interaction × size")

# Calculate effects at different size values
size_values = [0.5, 1.0, 1.5, 2.0]
h5_data = results_het.event_study[results_het.event_study['horizon'] == 5].iloc[0]
print("\nTreatment effect at horizon 5 for different sizes:")
for size in size_values:
    effect = h5_data['coefficient'] + h5_data['size_interaction'] * size
    print(f"  Size = {size}: Effect = {effect:.3f}")

# Example 4: Multiple interactions
print("\n" + "="*70)
print("Example 4: Multiple Interaction Terms")
print("="*70)

lpdid_multi_het = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ quality | industry",
    interactions="~ size + i.industry",  # Size and industry interactions
    n_jobs=-1
)

results_multi_het = lpdid_multi_het.fit(only_pooled=True)

print("\nPooled Results with Multiple Interactions:")
# Show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(results_multi_het.pooled)
pd.reset_option('display.max_columns')
pd.reset_option('display.width')

# Example 5: IV with heterogeneous effects
print("\n" + "="*70)
print("Example 5: IV + Heterogeneous Effects")
print("="*70)

lpdid_iv_het = LPDiD(
    data=df,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=5,
    post_window=10,
    formula="~ quality | industry | D_treat ~ eligible + spillover",
    interactions="~ size",
    wildbootstrap=499,
    n_jobs=-1
)

results_iv_het = lpdid_iv_het.fit()

print("\nIV + Heterogeneity Results (horizon 5):")
h5 = results_iv_het.event_study[results_iv_het.event_study['horizon'] == 5].iloc[0]
print(f"Base effect (size=0): {h5['coefficient']:.3f} (SE: {h5['se']:.3f})")
print(f"Size interaction: {h5['size_interaction']:.3f} (SE: {h5['size_interaction_se']:.3f})")
print(f"First-stage F-stat: {results_iv_het.iv_diagnostics[results_iv_het.iv_diagnostics['horizon']==5]['first_stage_F'].values[0]:.2f}")

# Example 6: Testing exclusion restriction
print("\n" + "="*70)
print("Example 6: Testing Exclusion Restriction")
print("="*70)

# Drop observations after treatment starts to test pre-trends in reduced form
df_pretreat = df[df['time'] < 15].copy()

print("\nReduced form pre-trends (instrument on outcome, pre-treatment only):")
# This tests whether instruments affect outcomes before treatment is available

lpdid_rf_test = LPDiD(
    data=df_pretreat,
    depvar='y',
    unit='unit',
    time='time',
    treat='eligible',  # Use instrument as "treatment"
    pre_window=5,
    post_window=0,
    formula="~ quality + size + spillover | industry",
    n_jobs=-1
)

results_rf_test = lpdid_rf_test.fit(only_event=True)
print("\nPre-treatment 'effects' of instrument (should be near zero):")
print(results_rf_test.event_study[['horizon', 'coefficient', 'se', 'p']])

# Example 7: Summary report
print("\n" + "="*70)
print("Example 7: Comprehensive Results Summary")
print("="*70)

# Use the summary method
print("\nFull IV + Heterogeneity Model Summary:")
results_iv_het.summary()

# Example 8: Comparing specifications
print("\n" + "="*70)
print("Example 8: Specification Comparison")
print("="*70)

specs = {
    'OLS': {
        'formula': '~ quality + size | industry'
    },
    'OLS + Oracle': {
        'formula': '~ quality + size + unobserved | industry'  # Cheating - using unobserved
    },
    'IV': {
        'formula': '~ quality + size | industry | D_treat ~ eligible + spillover'
    },
    'IV + Heterogeneity': {
        'formula': '~ quality | industry | D_treat ~ eligible + spillover',
        'interactions': '~ size'
    }
}

comparison = []
for name, spec in specs.items():
    print(f"\nRunning: {name}")
    
    lpdid_spec = LPDiD(
        data=df,
        depvar='y',
        unit='unit',
        time='time',
        treat='treat',
        pre_window=5,
        post_window=10,
        wildbootstrap=499 if 'IV' in name else None,
        n_jobs=-1,
        **spec
    )
    
    res = lpdid_spec.fit(only_pooled=True)
    pooled = res.pooled[res.pooled['period'] == 'Post'].iloc[0]
    
    comp_res = {
        'Specification': name,
        'Coefficient': pooled['coefficient'],
        'SE': pooled['se'],
        'P-value': pooled['p']
    }
    
    # Add IV diagnostics if applicable
    if 'first_stage_F' in pooled:
        comp_res['First-stage F'] = pooled['first_stage_F']
        comp_res['Weak IV'] = pooled['weak_iv']
    
    comparison.append(comp_res)

comparison_df = pd.DataFrame(comparison)
print("\n" + "="*70)
print("Specification Comparison (Pooled Post-treatment Effects):")
print("="*70)
print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

print("\nNotes:")
print("- OLS is biased upward due to positive correlation between treatment and unobservables")
print("- Oracle specification shows what we'd get if we could control for unobservables")
print("- IV corrects for endogeneity, getting closer to true effect")
print("- True average effect is around 2.0 + 1.5 * E[size] ≈ 3.25")