#!/usr/bin/env python3
"""
Test script to verify that the Polars fork() warning fix works correctly.

This script creates a simple dataset and runs LPDiD to ensure no Polars warnings appear.
Run this on Linux to test the fix.
"""

import warnings
import sys
import platform
import multiprocessing

print("="*60)
print("Testing Polars fork() Warning Fix")
print("="*60)
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Python version: {sys.version}")
print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")
print("="*60)

# Capture all warnings to check for Polars warnings
warnings.simplefilter("always")
captured_warnings = []

def warning_handler(message, category, filename, lineno, file=None, line=None):
    captured_warnings.append({
        'message': str(message),
        'category': category.__name__,
        'filename': filename,
        'lineno': lineno
    })
    # Also print the warning
    print(f"WARNING: {category.__name__}: {message}")

# Set up warning handler
old_showwarning = warnings.showwarning
warnings.showwarning = warning_handler

try:
    # Import pandas first (which may use Polars internally in some configurations)
    import pandas as pd
    import numpy as np
    
    print("\nImporting LPDiD...")
    from LPDiD import LPDiD
    
    print("✓ LPDiD imported successfully")
    
    # Create simple test data
    print("\nCreating test dataset...")
    np.random.seed(42)
    n_units = 100
    n_periods = 10
    
    # Create panel data
    data = []
    for unit in range(n_units):
        for period in range(n_periods):
            treat = 1 if unit < 50 and period >= 5 else 0
            y = np.random.normal(0, 1) + (2 * treat if treat else 0)
            data.append({
                'unit': unit,
                'time': period,
                'treat': treat,
                'y': y
            })
    
    df = pd.DataFrame(data)
    print(f"✓ Created dataset with {len(df)} observations")
    
    # Test LPDiD with multiprocessing
    print(f"\nTesting LPDiD with n_jobs=2...")
    
    try:
        lpdid = LPDiD(
            data=df,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=3,
            post_window=2,
            n_jobs=2  # Use multiprocessing
        )
        
        print("✓ LPDiD initialization successful")
        
        # Fit the model
        print("Running estimation...")
        results = lpdid.fit()
        
        print("✓ LPDiD estimation completed successfully")
        print(f"✓ Estimated {len(results.event_study)} horizons")
        
    except Exception as e:
        print(f"✗ LPDiD failed: {e}")
        sys.exit(1)
    
    # Check for Polars-related warnings
    print("\n" + "="*60)
    print("Warning Analysis")
    print("="*60)
    
    polars_warnings = [w for w in captured_warnings 
                      if 'polars' in w['message'].lower() or 'fork' in w['message'].lower()]
    
    if polars_warnings:
        print(f"✗ Found {len(polars_warnings)} Polars/fork-related warning(s):")
        for w in polars_warnings:
            print(f"  - {w['category']}: {w['message']}")
        print("\n⚠️  Fix may not be working correctly")
    else:
        print("✓ No Polars/fork-related warnings detected")
        print("✓ Fix is working correctly!")
    
    total_warnings = len(captured_warnings)
    print(f"\nTotal warnings captured: {total_warnings}")
    
    if total_warnings > 0:
        print("\nAll captured warnings:")
        for i, w in enumerate(captured_warnings, 1):
            print(f"  {i}. {w['category']}: {w['message']}")
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    if platform.system() == 'Linux':
        if multiprocessing.get_start_method() == 'spawn':
            print("✓ Running on Linux with 'spawn' start method")
        else:
            print(f"⚠️  Running on Linux but start method is '{multiprocessing.get_start_method()}'")
    else:
        print(f"ℹ️  Running on {platform.system()} (fix is Linux-specific)")
    
    print("✓ LPDiD multiprocessing test completed successfully")
    
    if not polars_warnings:
        print("✅ No Polars fork() warnings detected - Fix working!")
    else:
        print("❌ Polars fork() warnings still present - Fix needs review")
        sys.exit(1)

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure all required packages are installed:")
    print("  - pandas")
    print("  - numpy") 
    print("  - LPDiD")
    sys.exit(1)

except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    # Restore original warning handler
    warnings.showwarning = old_showwarning
