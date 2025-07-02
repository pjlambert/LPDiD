#!/usr/bin/env python3
"""
Test script to verify LP-DiD fixes
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Test 1: Check if the module imports without errors
print("Test 1: Importing LPDiD module...")
try:
    from LPDiD import LPDiD
    print("✓ Module imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Create simple test data
print("\nTest 2: Creating test data...")
np.random.seed(42)

# Create a simple panel dataset
n_units = 100
n_periods = 10
units = []
periods = []
treatment = []
outcome = []

for i in range(n_units):
    treated_unit = i < 50  # First 50 units are treated
    treatment_start = 6 if treated_unit else 999  # Treatment starts at period 6
    
    for t in range(n_periods):
        units.append(i)
        periods.append(t)
        
        # Treatment indicator
        treat = 1 if treated_unit and t >= treatment_start else 0
        treatment.append(treat)
        
        # Simple outcome with treatment effect
        y = np.random.normal(0, 1)
        if treat:
            y += 2.0  # Treatment effect of 2
        outcome.append(y)

test_data = pd.DataFrame({
    'unit_id': units,
    'time_id': periods,
    'treatment': treatment,
    'outcome': outcome
})

print(f"✓ Created test data with {len(test_data)} observations")

# Test 3: Test single clustering (should work)
print("\nTest 3: Testing single clustering...")
try:
    lpdid_single = LPDiD(
        data=test_data,
        depvar='outcome',
        unit='unit_id',
        time='time_id',
        treat='treatment',
        pre_window=3,
        post_window=3,
        cluster_formula='~ unit_id',  # Single clustering
        n_jobs=1,  # Use single core to avoid multiprocessing issues
        mp_type='spawn'  # Explicitly set spawn
    )
    
    results_single = lpdid_single.fit()
    print("✓ Single clustering works correctly")
    print(f"  - Estimated {len(results_single.event_study)} horizons")
except Exception as e:
    print(f"✗ Single clustering failed: {e}")

# Test 4: Test two-way clustering (this was failing before)
print("\nTest 4: Testing two-way clustering...")
try:
    lpdid_twoway = LPDiD(
        data=test_data,
        depvar='outcome',
        unit='unit_id',
        time='time_id',
        treat='treatment',
        pre_window=3,
        post_window=3,
        cluster_formula='~ unit_id + time_id',  # Two-way clustering
        n_jobs=1,  # Use single core to avoid multiprocessing issues
        mp_type='spawn'  # Explicitly set spawn
    )
    
    results_twoway = lpdid_twoway.fit()
    print("✓ Two-way clustering works correctly")
    print(f"  - Estimated {len(results_twoway.event_study)} horizons")
    
    # Show some results
    print("\n  Event study results:")
    print(results_twoway.event_study[['horizon', 'coefficient', 'se', 'p']])
    
except Exception as e:
    print(f"✗ Two-way clustering failed: {e}")

# Test 5: Test parallel processing with spawn
print("\nTest 5: Testing parallel processing with spawn...")
try:
    lpdid_parallel = LPDiD(
        data=test_data,
        depvar='outcome',
        unit='unit_id',
        time='time_id',
        treat='treatment',
        pre_window=3,
        post_window=3,
        cluster_formula='~ unit_id + time_id',
        n_jobs=2,  # Use 2 cores
        mp_type='spawn'  # Explicitly set spawn to avoid Polars fork warning
    )
    
    results_parallel = lpdid_parallel.fit()
    print("✓ Parallel processing with spawn works correctly")
    print(f"  - Estimated {len(results_parallel.event_study)} horizons")
    
except Exception as e:
    print(f"✗ Parallel processing failed: {e}")

print("\n" + "="*60)
print("Test Summary:")
print("="*60)
print("The main issue (vcov dict value must be a string) should now be fixed.")
print("Two-way clustering should work correctly with the string format.")
print("\nNote: The Polars fork() warning may still appear when using joblib,")
print("even with mp_type='spawn'. This is a warning, not an error.")
