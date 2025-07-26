#%%
import pandas as pd
import numpy as np
from LPDiD import LPDiD

# Set seed for reproducibility
np.random.seed(42)

#%%
# Create simple synthetic dataset (100 rows)
data = []
for unit in range(1, 11):  # 10 units
    for time in range(1, 11):  # 10 time periods
        # Simple treatment: units 6-10 get treated at time 6
        treat = 1 if unit >= 6 and time >= 6 else 0
        
        # Simple outcome with treatment effect
        outcome = 10 + 0.5 * time + (2 * treat) + np.random.normal(0, 1)
        
        # Control variable for min_time_selection demo
        employed = 1 if (unit + time) % 3 != 0 else 0  # ~67% employed
        
        # Another control variable
        sector = unit % 3  # 3 sectors
        
        data.append({
            'unit_id': unit,
            'time': time, 
            'treat': treat,
            'outcome': outcome,
            'employed': employed,
            'sector': sector
        })

df = pd.DataFrame(data)
print(f"Created synthetic dataset: {len(df)} rows, {df['unit_id'].nunique()} units, {df['time'].nunique()} periods")
print(f"Treatment starts at time 6 for units 6-10")
print(f"Employed rate: {df['employed'].mean():.1%}")
print()

#%%
# Configuration 1: Default settings
print("="*60)
print("1. DEFAULT SETTINGS")
print("="*60)
lpdid1 = LPDiD(
    data=df,
    depvar='outcome', 
    unit='unit_id',
    time='time',
    treat='treat', 
    pre_window=3,
    post_window=2,
    formula='~ sector'
)
lpdid1.build()
data1 = lpdid1.get_long_diff_data()
print(f"Result: {len(data1)} observations across {data1['h'].nunique()} horizons")
print()

#%%
# Configuration 2: min_time_selection only
print("="*60) 
print("2. MIN_TIME_SELECTION ONLY")
print("="*60)
lpdid2 = LPDiD(
    data=df,
    depvar='outcome',
    unit='unit_id', 
    time='time',
    treat='treat',
    pre_window=3,
    post_window=2,
    formula='~ sector',
    min_time_selection='employed == 1'
)
lpdid2.build()
data2 = lpdid2.get_long_diff_data()
print(f"Result: {len(data2)} observations across {data2['h'].nunique()} horizons")
print()

#%%
# Configuration 3: min_time_controls only
print("="*60)
print("3. MIN_TIME_CONTROLS ONLY") 
print("="*60)
lpdid3 = LPDiD(
    data=df,
    depvar='outcome',
    unit='unit_id',
    time='time', 
    treat='treat',
    pre_window=3,
    post_window=2,
    formula='~ sector',
    min_time_controls=True
)
lpdid3.build()
data3 = lpdid3.get_long_diff_data()
print(f"Result: {len(data3)} observations across {data3['h'].nunique()} horizons")
print()

#%%
# Configuration 4: Both together
print("="*60)
print("4. BOTH MIN_TIME_SELECTION AND MIN_TIME_CONTROLS")
print("="*60)
lpdid4 = LPDiD(
    data=df,
    depvar='outcome',
    unit='unit_id',
    time='time',
    treat='treat', 
    pre_window=3,
    post_window=2,
    formula='~ sector',
    min_time_selection='employed == 1',
    min_time_controls=True
)
lpdid4.build()
data4 = lpdid4.get_long_diff_data()
print(f"Result: {len(data4)} observations across {data4['h'].nunique()} horizons")
print()

#%%
# Simple comparison
print("="*60)
print("SUMMARY COMPARISON")
print("="*60)
print(f"1. Default:           {len(data1):,} observations")
print(f"2. Selection only:    {len(data2):,} observations") 
print(f"3. Controls only:     {len(data3):,} observations")
print(f"4. Both:              {len(data4):,} observations")
print()

# Show sample of differences for horizon 0
print("Sample data for horizon 0 (treatment period):")
print("Default (first 3 rows):")
print(data1[data1['h']==0][['unit_id', 'time', 'Dy', 'sector', 'employed']].head(3))
print()
print("With min_time_selection (first 3 rows):")
print(data2[data2['h']==0][['unit_id', 'time', 'Dy', 'sector', 'employed']].head(3))
