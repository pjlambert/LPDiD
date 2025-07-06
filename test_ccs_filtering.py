"""
Test script to verify CCS filtering is working correctly
"""

import numpy as np
import pandas as pd
from LPDiD import LPDiD

# Create a simple test dataset
np.random.seed(42)

# Create panel data
n_units = 100
n_periods = 10

# Create unit and time identifiers
unit_ids = []
time_ids = []
for i in range(n_units):
    for t in range(n_periods):
        unit_ids.append(i)
        time_ids.append(t)

# Create treatment indicator
# Units 0-19: never treated
# Units 20-39: treated at t=5
# Units 40-59: treated at t=6
# Units 60-79: treated at t=7
# Units 80-99: treated at t=8
treat = []
for i in range(n_units):
    for t in range(n_periods):
        if i < 20:
            treat.append(0)  # Never treated
        elif i < 40:
            treat.append(1 if t >= 5 else 0)  # Treated at t=5
        elif i < 60:
            treat.append(1 if t >= 6 else 0)  # Treated at t=6
        elif i < 80:
            treat.append(1 if t >= 7 else 0)  # Treated at t=7
        else:
            treat.append(1 if t >= 8 else 0)  # Treated at t=8

# Create outcome variable (with some treatment effect)
y = []
for i in range(n_units):
    for t in range(n_periods):
        base = i/10 + t + np.random.normal(0, 0.5)
        if treat[i * n_periods + t] == 1:
            base += 2.0  # Treatment effect
        y.append(base)

# Create DataFrame
data = pd.DataFrame({
    'unit': unit_ids,
    'time': time_ids,
    'treat': treat,
    'y': y
})

print("Test Dataset Summary:")
print(f"Total observations: {len(data)}")
print(f"Units: {data['unit'].nunique()}")
print(f"Time periods: {data['time'].nunique()}")
print(f"Never treated units: {(data.groupby('unit')['treat'].max() == 0).sum()}")
print(f"Treated units: {(data.groupby('unit')['treat'].max() == 1).sum()}")
print()

# Test 1: Standard CCS filtering
print("=" * 60)
print("Test 1: Standard CCS filtering")
print("=" * 60)

lpdid1 = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=3,
    post_window=2
)

# Build the data
lpdid1.build()

# Get the long difference data
long_data1 = lpdid1.get_long_diff_data()

print(f"\nLong difference data shape: {long_data1.shape}")
print(f"Unique horizons: {sorted(long_data1['h'].unique())}")

# Check CCS filtering for specific horizons
for h in sorted(long_data1['h'].unique()):
    h_data = long_data1[long_data1['h'] == h]
    treated = h_data[h_data['D_treat'] == 1]
    controls = h_data[h_data['D_treat'] == 0]
    
    print(f"\nHorizon {h}:")
    print(f"  Total obs: {len(h_data)}")
    print(f"  Newly treated: {len(treated)}")
    print(f"  Clean controls: {len(controls)}")
    
    # For controls, check which units they come from
    if len(controls) > 0:
        control_units = controls['unit'].unique()
        # Check if any of these units get treated later
        units_treated_later = []
        for u in control_units:
            unit_data = data[data['unit'] == u]
            max_treat_time = unit_data[unit_data['treat'] == 1]['time'].min() if any(unit_data['treat'] == 1) else np.inf
            # For this observation at time t, check if unit is treated by t+h
            control_obs = controls[controls['unit'] == u]
            for _, obs in control_obs.iterrows():
                t = obs['time']
                t_plus_h = t + h
                if max_treat_time <= t_plus_h:
                    units_treated_later.append(u)
                    break
        
        print(f"  Control units that get treated by t+h: {len(set(units_treated_later))}")
        if len(set(units_treated_later)) > 0:
            print(f"    WARNING: Found control units that should have been excluded!")

# Test 2: CCS filtering with nevertreated option
print("\n" + "=" * 60)
print("Test 2: CCS filtering with nevertreated=True")
print("=" * 60)

lpdid2 = LPDiD(
    data=data,
    depvar='y',
    unit='unit',
    time='time',
    treat='treat',
    pre_window=3,
    post_window=2,
    nevertreated=True
)

# Build the data
lpdid2.build()

# Get the long difference data
long_data2 = lpdid2.get_long_diff_data()

print(f"\nLong difference data shape: {long_data2.shape}")

# Check that all control observations come from never-treated units
control_obs = long_data2[long_data2['D_treat'] == 0]
control_units = control_obs['unit'].unique()

never_treated_units = data.groupby('unit')['treat'].max() == 0
never_treated_unit_ids = never_treated_units[never_treated_units].index.tolist()

print(f"\nControl units in data: {len(control_units)}")
print(f"Never treated units: {len(never_treated_unit_ids)}")
print(f"All control units are never-treated: {set(control_units).issubset(set(never_treated_unit_ids))}")

# Test 3: Compare filtering results
print("\n" + "=" * 60)
print("Test 3: Comparison of filtering results")
print("=" * 60)

print(f"\nStandard CCS filtering:")
print(f"  Total observations: {len(long_data1)}")
print(f"  Treated observations: {len(long_data1[long_data1['D_treat'] == 1])}")
print(f"  Control observations: {len(long_data1[long_data1['D_treat'] == 0])}")

print(f"\nWith nevertreated=True:")
print(f"  Total observations: {len(long_data2)}")
print(f"  Treated observations: {len(long_data2[long_data2['D_treat'] == 1])}")
print(f"  Control observations: {len(long_data2[long_data2['D_treat'] == 0])}")

print(f"\nReduction in control observations: {len(long_data1[long_data1['D_treat'] == 0]) - len(long_data2[long_data2['D_treat'] == 0])}")

print("\nTest complete!")
