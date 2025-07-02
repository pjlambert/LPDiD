"""
Test script to demonstrate the mp_type parameter functionality for LPDiD and LPDiDPois.
This addresses the Polars fork() warning on Linux systems.
"""

import pandas as pd
import numpy as np
from LPDiD import LPDiD, LPDiDPois
import multiprocessing
import platform

def create_test_data(n_units=50, n_periods=10):
    """Create simple test data for demonstration"""
    # Create panel data
    units = list(range(1, n_units + 1))
    periods = list(range(1, n_periods + 1))
    
    data = []
    for unit in units:
        for period in periods:
            # Simple treatment assignment: units 26-50 get treated in period 6
            treat = 1 if unit > n_units//2 and period >= 6 else 0
            
            # Simple outcome with treatment effect
            y = 10 + 0.5 * unit + 0.2 * period + 2 * treat + np.random.normal(0, 1)
            
            data.append({
                'unit': unit,
                'time': period,
                'y': y,
                'treat': treat
            })
    
    return pd.DataFrame(data)

def test_mp_type_parameter():
    """Test the mp_type parameter functionality"""
    print("="*60)
    print("Testing mp_type parameter functionality")
    print("="*60)
    
    print(f"System: {platform.system()}")
    print(f"Current multiprocessing start method: {multiprocessing.get_start_method()}")
    
    # Create test data
    data = create_test_data()
    print(f"\nCreated test data with {len(data)} observations")
    
    # Test 1: LPDiD with mp_type='spawn' (recommended for Linux)
    print("\n" + "-"*40)
    print("Test 1: LPDiD with mp_type='spawn'")
    print("-"*40)
    
    try:
        model = LPDiD(
            data=data,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=2,
            post_window=2,
            n_jobs=2,  # Use 2 cores to test parallel functionality
            mp_type='spawn'  # This should prevent Polars fork() warnings
        )
        
        print("✓ LPDiD initialized successfully with mp_type='spawn'")
        print(f"Current multiprocessing method after initialization: {multiprocessing.get_start_method()}")
        
        # Run estimation (this would trigger parallel processing in step 4)
        results = model.fit()
        print("✓ LPDiD estimation completed successfully")
        
    except Exception as e:
        print(f"✗ LPDiD with mp_type='spawn' failed: {e}")
    
    # Test 2: LPDiDPois with mp_type='spawn'
    print("\n" + "-"*40)
    print("Test 2: LPDiDPois with mp_type='spawn'")
    print("-"*40)
    
    try:
        # Create count data for Poisson
        count_data = data.copy()
        count_data['y'] = np.maximum(0, count_data['y'].round().astype(int))
        
        model_pois = LPDiDPois(
            data=count_data,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=2,
            post_window=2,
            n_jobs=2,
            mp_type='spawn'
        )
        
        print("✓ LPDiDPois initialized successfully with mp_type='spawn'")
        
        # Run estimation
        results_pois = model_pois.fit()
        print("✓ LPDiDPois estimation completed successfully")
        
    except Exception as e:
        print(f"✗ LPDiDPois with mp_type='spawn' failed: {e}")
    
    # Test 3: Test invalid mp_type
    print("\n" + "-"*40)
    print("Test 3: Invalid mp_type parameter")
    print("-"*40)
    
    try:
        model_invalid = LPDiD(
            data=data,
            depvar='y',
            unit='unit',
            time='time',
            treat='treat',
            pre_window=2,
            post_window=2,
            mp_type='invalid_method'
        )
        print("✗ Should have failed with invalid mp_type")
    except ValueError as e:
        print(f"✓ Correctly caught invalid mp_type: {e}")
    except Exception as e:
        print(f"? Unexpected error with invalid mp_type: {e}")
    
    print("\n" + "="*60)
    print("mp_type parameter testing completed")
    print("="*60)

if __name__ == "__main__":
    test_mp_type_parameter()
