# LP-DiD Quick Start Guide

## Installation

```bash
pip install numpy pandas pyfixest scipy joblib matplotlib
# Then install lpdid (from source for now)
git clone https://github.com/yourusername/lpdid.git
cd lpdid
pip install -e .
```

## Basic Example (5 minutes)

```python
import pandas as pd
from lpdid import LPDiD

# Load your data
df = pd.read_csv('your_panel_data.csv')

# Run LP-DiD
lpdid = LPDiD(
    data=df,
    depvar='outcome',
    unit='firm_id',
    time='year',
    treat='treated',
    pre_window=3,
    post_window=5,
    formula="~ control1 + control2 | industry",
    n_jobs=-1  # Use all CPU cores
)

# Get results
results = lpdid.fit()

# View event study
print(results.event_study)

# View pooled effects
print(results.pooled)

# Get summary
results.summary()
```

## Common Use Cases

### 1. Simple DiD with Controls
```python
lpdid = LPDiD(
    data=df,
    depvar='revenue',
    unit='firm_id',
    time='quarter',
    treat='policy',
    pre_window=4,
    post_window=8,
    formula="~ size + age | industry + region"
)
results = lpdid.fit()
```

### 2. IV for Endogenous Treatment
```python
# When treatment might be endogenous
lpdid_iv = LPDiD(
    data=df,
    depvar='employment',
    unit='firm_id',
    time='year',
    treat='subsidy_received',
    pre_window=3,
    post_window=5,
    formula="~ size | industry | D_treat ~ eligible + neighbor_treated"
)
results = lpdid_iv.fit()

# Check if instruments are strong
print(results.iv_diagnostics)  # Look for F-stat > 10
```

### 3. Heterogeneous Treatment Effects
```python
# How do effects vary by firm characteristics?
lpdid_het = LPDiD(
    data=df,
    depvar='productivity',
    unit='firm_id',
    time='year',
    treat='training_program',
    pre_window=2,
    post_window=4,
    formula="~ baseline_prod | industry",
    interactions="~ size + tech_intensity"
)
results = lpdid_het.fit()

# Check interaction terms
print(results.event_study[['horizon', 'coefficient', 'size_interaction']])
```

### 4. Robust Standard Errors
```python
# Wild bootstrap for better inference
lpdid_robust = LPDiD(
    data=df,
    depvar='wages',
    unit='firm_id',
    time='year',
    treat='min_wage_increase',
    pre_window=3,
    post_window=5,
    formula="~ employment | industry + county",
    cluster_formula="~ state",  # Cluster at state level
    wildbootstrap=999,  # Wild bootstrap iterations
    seed=123  # For reproducibility
)
results = lpdid_robust.fit()
```

## Interpreting Results

### Event Study Output
```
horizon  coefficient    se      p     ci_low   ci_high   obs
-3       0.021        0.045   0.642  -0.067   0.109     1200
-2       0.015        0.038   0.694  -0.059   0.089     1250
-1       0.000        0.000   NaN     0.000   0.000     1300
0        0.234        0.072   0.001   0.093   0.375     1300
1        0.456        0.084   0.000   0.291   0.621     1250
2        0.523        0.091   0.000   0.345   0.701     1200
```

- **Pre-treatment** (negative horizons): Should be near zero (parallel trends)
- **Horizon -1**: Reference period (always zero)
- **Post-treatment** (positive horizons): Treatment effects over time

### IV Diagnostics
```
horizon  first_stage_F  weak_iv
0        45.3          False
1        43.7          False
2        41.2          False
```

- **first_stage_F > 10**: Instruments are strong
- **weak_iv = True**: Warning for weak instruments

### Heterogeneous Effects
```
horizon  coefficient  size_interaction  size_interaction_p
0        0.150       0.082            0.023
1        0.289       0.134            0.001
2        0.356       0.156            0.000
```

- **coefficient**: Effect for baseline (size = 0)
- **size_interaction**: How effect changes with size
- **Total effect** = coefficient + size_interaction × size_value

## Tips

1. **Start simple**: Run without IV/interactions first
2. **Check pre-trends**: Look at negative horizons
3. **Use parallel**: Set `n_jobs=-1` for speed
4. **Bootstrap iterations**: Start with 499, increase for final results
5. **Save results**: `results_df = results.event_study.copy()`

## Common Issues

### "Regression failed"
- Check for sufficient variation in treatment
- Ensure panel is balanced enough
- Try fewer horizons or controls

### Weak instruments (F < 10)
- Find stronger instruments
- Consider OLS as robustness check
- Report both OLS and IV results

### Slow performance
- Reduce horizons
- Use fewer bootstrap iterations initially
- Ensure `n_jobs=-1` for parallelization

## Need Help?

- Full documentation: See README.md
- Examples: Check examples/ directory
- Issues: GitHub issues page