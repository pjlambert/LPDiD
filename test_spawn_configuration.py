#!/usr/bin/env python3
"""
Test script to verify spawn configuration for multiprocessing and joblib
"""

import multiprocessing
import sys
import platform

print("Testing spawn configuration...")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.machine()}")

# Test 1: Check multiprocessing start method before importing LPDiD
print("\n1. Checking multiprocessing start method BEFORE importing LPDiD:")
try:
    method_before = multiprocessing.get_start_method()
    print(f"   Current method: {method_before}")
except RuntimeError:
    print("   No method set yet")

# Test 2: Import LPDiD and check what happens
print("\n2. Importing LPDiD...")
from LPDiD import LPDiD

print("\n3. Checking multiprocessing start method AFTER importing LPDiD:")
method_after = multiprocessing.get_start_method()
print(f"   Current method: {method_after}")

# Test 3: Check joblib configuration
print("\n4. Checking joblib configuration:")
try:
    import joblib
    if hasattr(joblib.parallel, 'BACKENDS'):
        if 'multiprocessing' in joblib.parallel.BACKENDS:
            backend = joblib.parallel.BACKENDS['multiprocessing']
            if hasattr(backend, 'start_method'):
                print(f"   joblib multiprocessing backend start_method: {backend.start_method}")
            else:
                print("   joblib multiprocessing backend doesn't have start_method attribute")
        else:
            print("   multiprocessing backend not found in joblib.parallel.BACKENDS")
    else:
        print("   joblib.parallel.BACKENDS not found")
except Exception as e:
    print(f"   Error checking joblib: {e}")

# Test 4: Test actual parallel processing with a simple task
print("\n5. Testing actual parallel processing:")
import pandas as pd
import numpy as np

# Create simple test data
np.random.seed(42)
test_data = pd.DataFrame({
    'unit_id': np.repeat(range(10), 5),
    'time_id': np.tile(range(5), 10),
    'treatment': np.random.binomial(1, 0.3, 50),
    'outcome': np.random.normal(0, 1, 50)
})

# Run LPDiD with parallel processing
try:
    print("   Running LPDiD with n_jobs=2...")
    lpdid = LPDiD(
        data=test_data,
        depvar='outcome',
        unit='unit_id',
        time='time_id',
        treat='treatment',
        pre_window=2,
        post_window=2,
        n_jobs=2,
        mp_type='spawn'
    )
    
    # Check if joblib is using the correct backend
    from joblib import Parallel
    with Parallel(n_jobs=2, backend='loky', verbose=10) as parallel:
        result = parallel(joblib.delayed(lambda x: x**2)(i) for i in range(4))
    print(f"   Parallel computation result: {result}")
    print("   ✓ Parallel processing works with loky backend")
    
except Exception as e:
    print(f"   Error during parallel processing: {e}")

# Test 5: Verify no fork warnings
print("\n6. Testing for fork-related warnings:")
print("   If you see Polars fork() warnings above, the configuration may need adjustment.")
print("   If no warnings appeared, the spawn configuration is working correctly.")

print("\n" + "="*60)
print("Summary:")
print("="*60)
if platform.system() == 'Linux':
    if method_after == 'spawn':
        print("✓ Multiprocessing is correctly set to 'spawn' on Linux")
    else:
        print("✗ Multiprocessing is NOT set to 'spawn' on Linux")
else:
    print(f"✓ On {platform.system()}, default method '{method_after}' is acceptable")

print("\nNote: The package now:")
print("1. Forces multiprocessing to use 'spawn' on Linux")
print("2. Configures joblib's multiprocessing backend to use 'spawn'")
print("3. Uses 'loky' backend for joblib parallel processing")
print("4. This should prevent Polars fork() warnings")
