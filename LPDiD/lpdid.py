"""
Local Projections Difference-in-Differences (LP-DiD) for Python
Based on Dube, Girardi, Jordà, and Taylor (2023)
"""

# Set OpenMP environment variables BEFORE importing numpy/scipy
import os

# Handle OpenMP configuration for high-performance scenarios
# Only set if user hasn't explicitly configured these
if 'OMP_NESTED' not in os.environ:
    # Disable nested parallelism to avoid deprecated warning
    os.environ['OMP_NESTED'] = 'FALSE'
    
if 'OMP_MAX_ACTIVE_LEVELS' not in os.environ:
    # Set max active levels to 1 to prevent nesting
    os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'

# IMPORTANT: We do NOT set OMP_NUM_THREADS here
# This allows:
# 1. Users to set it themselves for their specific hardware
# 2. The system to use all available cores by default
# 3. Optimal performance on high-core-count systems (e.g., 64 cores)

# For best performance with Ray parallelism, users can set:
# - OMP_NUM_THREADS=1 when using many Ray workers (n_jobs=-1)
# - OMP_NUM_THREADS=(cores/n_jobs) for balanced thread/process parallelism

import warnings
# Suppress various warnings that can occur during parallel processing
warnings.filterwarnings("ignore", message=".*omp_set_nested.*")
warnings.filterwarnings("ignore", message=".*OMP.*")
warnings.filterwarnings("ignore", message=".*A worker stopped while some jobs were given to the executor.*")
# Suppress Polars fork warnings
warnings.filterwarnings("ignore", message=".*fork.*")
warnings.filterwarnings("ignore", message=".*Polars.*fork.*")
warnings.filterwarnings("ignore", message=".*using fork.*")

# Set environment variable to disable Polars fork warnings
if 'POLARS_WARN_UNSTABLE_FORK' not in os.environ:
    os.environ['POLARS_WARN_UNSTABLE_FORK'] = '0'

import sys
import platform
import multiprocessing

# Note: We do NOT force multiprocessing configuration at module import time
# This avoids conflicts with other packages and user configurations
# Instead, configuration happens within the LPDiD class when mp_type is specified

import numpy as np
import pandas as pd
import pyfixest as pf
from typing import Optional, Union, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import ray
from dataclasses import dataclass
from scipy import stats
import re
from tqdm import tqdm
import json

# Import wildboottest package
try:
    from wildboottest.wildboottest import wildboottest
    WILDBOOTTEST_AVAILABLE = True
except ImportError:
    # We'll use pyfixest's built-in wildboottest method instead
    WILDBOOTTEST_AVAILABLE = False


# Ray remote function for parallel regression
@ray.remote
def _run_single_regression_ray(
    horizon, is_pre, long_diff_data, depvar, time, controls, absorb, 
    cluster_vars, interact_vars, wildbootstrap, seed, weights, 
    lean, copy_data
):
    """
    Ray remote function to run a single regression for a specific horizon.
    
    This function is decorated with @ray.remote to run in parallel across
    multiple Ray workers. It contains all the logic from the original
    _run_single_regression method but as a standalone function.
    """
    # Determine horizon value for filtering
    horizon_value = -horizon if is_pre else horizon
    
    # Filter long format data for this horizon (already CCS filtered)
    reg_data = long_diff_data[long_diff_data['h'] == horizon_value].copy()
    
    if reg_data.shape[0] == 0:
        return None
    
    # Build formula using 'Dy' as the dependent variable
    # Ensure unique fixed effects (avoid duplicates)
    fe_vars = [time] + absorb
    fe_vars = list(dict.fromkeys(fe_vars))  # Remove duplicates while preserving order
    formula = _build_regression_formula_standalone('Dy', fe_vars, controls, interact_vars, long_diff_data, time)
    
    # Set up clustering
    if len(cluster_vars) == 1:
        vcov = {'CRV1': cluster_vars[0]}
    elif len(cluster_vars) == 2:
        # Two-way clustering - use CRV1 with string format
        vcov = {'CRV1': ' + '.join(cluster_vars)}
    else:
        # For 3+ clustering variables, also use string format
        vcov = {'CRV1': ' + '.join(cluster_vars)}
    
    # Add weights if specified
    use_weights = weights if weights else None
    
    # Run the fixed-effects regression using pyfixest.
    try:
        if use_weights is not None:
            fit = pf.feols(
                formula,
                data=reg_data,
                weights=use_weights,
                vcov=vcov,
                lean=lean,
                copy_data=copy_data
            )
        else:
            fit = pf.feols(
                formula,
                data=reg_data,
                vcov=vcov,
                lean=lean,
                copy_data=copy_data
            )
        
        nobs = fit._N
        
        # Store results dictionary
        results = {
            'horizon': -horizon if is_pre else horizon,
            'obs': nobs
        }
        
        # Handle different cases: group-specific effects vs main effect
        if interact_vars:
            # With interactions: we have separate coefficients for each group
            # Extract coefficients for each group
            first_group_found = False
            
            for var in interact_vars:
                clean_var = var.replace("i.", "") if var.startswith("i.") else var
                unique_vals = sorted(long_diff_data[clean_var].dropna().unique())
                
                for val in unique_vals:
                    group_name = f"{clean_var}_{val}"
                    coef_name = f"D_treat_{group_name}"
                    
                    if coef_name in fit.coef().index:
                        group_coef = fit.coef().loc[coef_name]
                        
                        # Store group-specific results
                        results[f'{clean_var}_{val}_coef'] = group_coef
                        
                        # Extract standard errors and inference for this group
                        if hasattr(fit, 'se') and coef_name in fit.se().index:
                            results[f'{clean_var}_{val}_se'] = fit.se().loc[coef_name]
                        if hasattr(fit, 'tstat') and coef_name in fit.tstat().index:
                            results[f'{clean_var}_{val}_t'] = fit.tstat().loc[coef_name]
                        if hasattr(fit, 'pvalue') and coef_name in fit.pvalue().index:
                            results[f'{clean_var}_{val}_p'] = fit.pvalue().loc[coef_name]
                        if hasattr(fit, 'confint'):
                            try:
                                ci = fit.confint()
                                if coef_name in ci.index:
                                    ci_row = ci.loc[coef_name]
                                    if hasattr(ci_row, 'iloc'):
                                        results[f'{clean_var}_{val}_ci_low'] = ci_row.iloc[0]
                                        results[f'{clean_var}_{val}_ci_high'] = ci_row.iloc[1]
                                    else:
                                        results[f'{clean_var}_{val}_ci_low'] = ci_row[0]
                                        results[f'{clean_var}_{val}_ci_high'] = ci_row[1]
                            except Exception:
                                pass  # Skip CI if extraction fails
                        
                        # Use first group for backward compatibility main columns
                        if not first_group_found:
                            results['coefficient'] = group_coef
                            if hasattr(fit, 'se') and coef_name in fit.se().index:
                                results['se'] = fit.se().loc[coef_name]
                            if hasattr(fit, 'tstat') and coef_name in fit.tstat().index:
                                results['t'] = fit.tstat().loc[coef_name]
                            if hasattr(fit, 'pvalue') and coef_name in fit.pvalue().index:
                                results['p'] = fit.pvalue().loc[coef_name]
                            if hasattr(fit, 'confint'):
                                try:
                                    ci = fit.confint()
                                    if coef_name in ci.index:
                                        ci_row = ci.loc[coef_name]
                                        if hasattr(ci_row, 'iloc'):
                                            results['ci_low'] = ci_row.iloc[0]
                                            results['ci_high'] = ci_row.iloc[1]
                                        else:
                                            results['ci_low'] = ci_row[0]
                                            results['ci_high'] = ci_row[1]
                                except Exception:
                                    pass  # Skip CI if extraction fails
                            first_group_found = True
            
            # If no groups were found, set NaN values
            if not first_group_found:
                results['coefficient'] = np.nan
                results['se'] = np.nan
                results['t'] = np.nan
                results['p'] = np.nan
                results['ci_low'] = np.nan
                results['ci_high'] = np.nan
        else:
            # Without interactions: extract main D_treat coefficient only
            if 'D_treat' not in fit.coef().index:
                warnings.warn(f"D_treat coefficient not found for horizon {horizon}")
                return None
                
            # Extract main treatment coefficient
            coef = fit.coef().loc['D_treat']
            results['coefficient'] = coef
            
            # Handle inference for main effect
            if wildbootstrap:
                # Use pyfixest's built-in wildboottest method
                try:
                    # Set seed for this specific test if provided
                    if seed is not None:
                        np.random.seed(seed + horizon)  # Add horizon for variation
                    
                    wb_result = fit.wildboottest(
                        param="D_treat",
                        reps=wildbootstrap
                    )
                    
                    # Extract results from the wildboottest output (pandas Series)
                    if isinstance(wb_result, pd.Series):
                        # Extract wild bootstrap p-value (this is the main adjustment)
                        if 'Pr(>|t|)' in wb_result.index:
                            results['p'] = wb_result['Pr(>|t|)']
                        # Extract wild bootstrap t-statistic
                        if 't value' in wb_result.index:
                            results['t'] = wb_result['t value']
                    
                    # Use analytical standard errors and confidence intervals
                    if hasattr(fit, 'se') and 'D_treat' in fit.se().index:
                        results['se'] = fit.se().loc['D_treat']
                    if hasattr(fit, 'confint'):
                        try:
                            ci = fit.confint()
                            if 'D_treat' in ci.index:
                                ci_row = ci.loc['D_treat']
                                if hasattr(ci_row, 'iloc'):
                                    results['ci_low'] = ci_row.iloc[0]
                                    results['ci_high'] = ci_row.iloc[1]
                                else:
                                    results['ci_low'] = ci_row[0]
                                    results['ci_high'] = ci_row[1]
                        except Exception:
                            pass  # Skip CI if extraction fails
                        
                except Exception as e:
                    warnings.warn(f"Wild bootstrap failed, using analytical inference: {e}")
                    # Fall back to analytical standard errors
                    if hasattr(fit, 'se') and 'D_treat' in fit.se().index:
                        results['se'] = fit.se().loc['D_treat']
                    if hasattr(fit, 'tstat') and 'D_treat' in fit.tstat().index:
                        results['t'] = fit.tstat().loc['D_treat']
                    if hasattr(fit, 'pvalue') and 'D_treat' in fit.pvalue().index:
                        results['p'] = fit.pvalue().loc['D_treat']
                    if hasattr(fit, 'confint'):
                        try:
                            ci = fit.confint()
                            if 'D_treat' in ci.index:
                                ci_row = ci.loc['D_treat']
                                if hasattr(ci_row, 'iloc'):
                                    results['ci_low'] = ci_row.iloc[0]
                                    results['ci_high'] = ci_row.iloc[1]
                                else:
                                    results['ci_low'] = ci_row[0]
                                    results['ci_high'] = ci_row[1]
                        except Exception:
                            pass  # Skip CI if extraction fails
            else:
                # Use analytical standard errors
                if hasattr(fit, 'se') and 'D_treat' in fit.se().index:
                    results['se'] = fit.se().loc['D_treat']
                if hasattr(fit, 'tstat') and 'D_treat' in fit.tstat().index:
                    results['t'] = fit.tstat().loc['D_treat']
                if hasattr(fit, 'pvalue') and 'D_treat' in fit.pvalue().index:
                    results['p'] = fit.pvalue().loc['D_treat']
                if hasattr(fit, 'confint'):
                    try:
                        ci = fit.confint()
                        if 'D_treat' in ci.index:
                            ci_row = ci.loc['D_treat']
                            if hasattr(ci_row, 'iloc'):
                                results['ci_low'] = ci_row.iloc[0]
                                results['ci_high'] = ci_row.iloc[1]
                            else:
                                results['ci_low'] = ci_row[0]
                                results['ci_high'] = ci_row[1]
                    except Exception:
                        pass  # Skip CI if extraction fails
        
        return results
        
    except Exception as e:
        warnings.warn(f"Regression failed for horizon {horizon}: {e}")
        return None


# Ray remote function for Poisson regression
@ray.remote
def _run_single_regression_ray_poisson(
    horizon, is_pre, long_diff_data, depvar, time, controls, absorb, 
    cluster_vars, interact_vars, wildbootstrap, seed, weights, 
    lean, copy_data
):
    """
    Ray remote function to run a single Poisson regression for a specific horizon.
    
    This function is decorated with @ray.remote to run in parallel across
    multiple Ray workers. It contains all the logic from the original
    _run_single_regression method for Poisson regression.
    """
    # Determine horizon value for filtering
    horizon_value = -horizon if is_pre else horizon
    
    # Filter long format data for this horizon (already CCS filtered)
    reg_data = long_diff_data[long_diff_data['h'] == horizon_value].copy()
    
    if reg_data.shape[0] == 0:
        return None
    
    # For Poisson regression, we need non-negative outcomes
    # Check if we have negative values and warn
    if reg_data['Dy'].min() < 0:
        warnings.warn(f"Negative values detected in long-differenced outcome. "
                     "Consider using swap_pre_diff=True for pre-treatment periods "
                     "or use regular LPDiD for continuous outcomes.")
        # Convert negative values to 0 for Poisson
        reg_data['Dy'] = np.maximum(reg_data['Dy'], 0)
    
    # Build formula using 'Dy' as the dependent variable
    # Ensure unique fixed effects (avoid duplicates)
    fe_vars = [time] + absorb
    fe_vars = list(dict.fromkeys(fe_vars))  # Remove duplicates while preserving order
    formula = _build_regression_formula_standalone('Dy', fe_vars, controls, interact_vars, long_diff_data, time)
    
    # Set up clustering
    if len(cluster_vars) == 1:
        vcov = {'CRV1': cluster_vars[0]}
    elif len(cluster_vars) == 2:
        # Two-way clustering - use CRV1 with string format
        vcov = {'CRV1': ' + '.join(cluster_vars)}
    else:
        # For 3+ clustering variables, also use string format
        vcov = {'CRV1': ' + '.join(cluster_vars)}
    
    # Add weights if specified
    use_weights = weights if weights else None
    
    # Run Poisson regression using pyfixest.
    try:
        if use_weights is not None:
            fit = pf.fepois(
                formula,
                data=reg_data,
                weights=use_weights,
                vcov=vcov,
                lean=lean,
                copy_data=copy_data
            )
        else:
            fit = pf.fepois(
                formula,
                data=reg_data,
                vcov=vcov,
                lean=lean,
                copy_data=copy_data
            )
        
        nobs = fit._N
        
        # Store results dictionary
        results = {
            'horizon': -horizon if is_pre else horizon,
            'obs': nobs
        }
        
        # Handle different cases: group-specific effects vs main effect
        if interact_vars:
            # With interactions: we have separate coefficients for each group
            # Extract coefficients for each group
            first_group_found = False
            
            for var in interact_vars:
                clean_var = var.replace("i.", "") if var.startswith("i.") else var
                unique_vals = sorted(long_diff_data[clean_var].dropna().unique())
                
                for val in unique_vals:
                    group_name = f"{clean_var}_{val}"
                    coef_name = f"D_treat_{group_name}"
                    
                    if coef_name in fit.coef().index:
                        group_coef = fit.coef().loc[coef_name]
                        
                        # Store group-specific results
                        results[f'{clean_var}_{val}_coef'] = group_coef
                        
                        # Extract standard errors and inference for this group
                        if hasattr(fit, 'se') and coef_name in fit.se().index:
                            results[f'{clean_var}_{val}_se'] = fit.se().loc[coef_name]
                        if hasattr(fit, 'tstat') and coef_name in fit.tstat().index:
                            results[f'{clean_var}_{val}_t'] = fit.tstat().loc[coef_name]
                        if hasattr(fit, 'pvalue') and coef_name in fit.pvalue().index:
                            results[f'{clean_var}_{val}_p'] = fit.pvalue().loc[coef_name]
                        if hasattr(fit, 'confint'):
                            try:
                                ci = fit.confint()
                                if coef_name in ci.index:
                                    ci_row = ci.loc[coef_name]
                                    if hasattr(ci_row, 'iloc'):
                                        results[f'{clean_var}_{val}_ci_low'] = ci_row.iloc[0]
                                        results[f'{clean_var}_{val}_ci_high'] = ci_row.iloc[1]
                                    else:
                                        results[f'{clean_var}_{val}_ci_low'] = ci_row[0]
                                        results[f'{clean_var}_{val}_ci_high'] = ci_row[1]
                            except Exception:
                                pass  # Skip CI if extraction fails
                        
                        # Use first group for backward compatibility main columns
                        if not first_group_found:
                            results['coefficient'] = group_coef
                            if hasattr(fit, 'se') and coef_name in fit.se().index:
                                results['se'] = fit.se().loc[coef_name]
                            if hasattr(fit, 'tstat') and coef_name in fit.tstat().index:
                                results['t'] = fit.tstat().loc[coef_name]
                            if hasattr(fit, 'pvalue') and coef_name in fit.pvalue().index:
                                results['p'] = fit.pvalue().loc[coef_name]
                            if hasattr(fit, 'confint'):
                                try:
                                    ci = fit.confint()
                                    if coef_name in ci.index:
                                        ci_row = ci.loc[coef_name]
                                        if hasattr(ci_row, 'iloc'):
                                            results['ci_low'] = ci_row.iloc[0]
                                            results['ci_high'] = ci_row.iloc[1]
                                        else:
                                            results['ci_low'] = ci_row[0]
                                            results['ci_high'] = ci_row[1]
                                except Exception:
                                    pass  # Skip CI if extraction fails
                            first_group_found = True
            
            # If no groups were found, set NaN values
            if not first_group_found:
                results['coefficient'] = np.nan
                results['se'] = np.nan
                results['t'] = np.nan
                results['p'] = np.nan
                results['ci_low'] = np.nan
                results['ci_high'] = np.nan
        else:
            # Without interactions: extract main D_treat coefficient only
            if 'D_treat' not in fit.coef().index:
                warnings.warn(f"D_treat coefficient not found for horizon {horizon}")
                return None
                
            # Extract main treatment coefficient
            coef = fit.coef().loc['D_treat']
            results['coefficient'] = coef
            
            # Handle inference for main effect
            if wildbootstrap:
                # Note: Wild bootstrap for Poisson may not be as well-supported
                warnings.warn("Wild bootstrap with Poisson regression may have limitations. "
                             "Consider using analytical inference.")
            
            # Use analytical standard errors
            if hasattr(fit, 'se') and 'D_treat' in fit.se().index:
                results['se'] = fit.se().loc['D_treat']
            if hasattr(fit, 'tstat') and 'D_treat' in fit.tstat().index:
                results['t'] = fit.tstat().loc['D_treat']
            if hasattr(fit, 'pvalue') and 'D_treat' in fit.pvalue().index:
                results['p'] = fit.pvalue().loc['D_treat']
            if hasattr(fit, 'confint'):
                try:
                    ci = fit.confint()
                    if 'D_treat' in ci.index:
                        ci_row = ci.loc['D_treat']
                        if hasattr(ci_row, 'iloc'):
                            results['ci_low'] = ci_row.iloc[0]
                            results['ci_high'] = ci_row.iloc[1]
                        else:
                            results['ci_low'] = ci_row[0]
                            results['ci_high'] = ci_row[1]
                except Exception:
                    pass  # Skip CI if extraction fails
        
        return results
        
    except Exception as e:
        warnings.warn(f"Poisson regression failed for horizon {horizon}: {e}")
        return None


# Standalone helper functions for Ray remote function
def _build_regression_formula_standalone(y_var, fe_vars, controls, interact_vars, data, time):
    """Standalone version of _build_regression_formula for Ray"""
    # Build control variables part - always use cntr_ versions
    if interact_vars:
        # With interactions, we use group-specific treatment variables
        controls_list = []
        
        # Add cntr_ versions of control variables
        for ctrl in controls:
            controls_list.append(f"cntr_{ctrl}")
        
        # Add group-specific treatment indicators
        for var in interact_vars:
            clean_var = var.replace("i.", "") if var.startswith("i.") else var
            unique_vals = sorted(data[clean_var].dropna().unique())
            
            for val in unique_vals:
                group_name = f"{clean_var}_{val}"
                controls_list.append(f"D_treat_{group_name}")
    else:
        # Standard case: use cntr_ versions of controls and add D_treat
        controls_list = [f"cntr_{ctrl}" for ctrl in controls] + ['D_treat']
    
    # Build fixed effects part - use cntr_ versions except for time
    fe_cntr = []
    for fe in fe_vars:
        if fe == time:
            fe_cntr.append(fe)  # Time variable stays as is
        else:
            fe_cntr.append(f"cntr_{fe}")  # Other fixed effects use cntr_ version
    
    # Build formula parts
    controls_str = " + ".join(controls_list) if controls_list else "1"
    fe_str = " + ".join(fe_cntr) if fe_cntr else ""
    
    if fe_str:
        formula = f"{y_var} ~ {controls_str} | {fe_str}"
    else:
        formula = f"{y_var} ~ {controls_str}"
    
    return formula


@dataclass
class LPDiDResults:
    """Container for LP-DiD results"""
    event_study: Optional[pd.DataFrame] = None
    pooled: Optional[pd.DataFrame] = None
    depvar: str = ""
    pre_window: Optional[int] = None
    post_window: Optional[int] = None
    control_group: str = ""
    treated_group: str = ""
    
    def summary(self):
        """Print summary of results"""
        print("\n" + "="*60)
        print("LP-DiD Results Summary")
        print("="*60)
        
        print(f"\nDependent variable: {self.depvar}")
        print(f"Pre-treatment window: {self.pre_window}")
        print(f"Post-treatment window: {self.post_window}")
        print(f"Control group: {self.control_group}")
        print(f"Treated group: {self.treated_group}")
        
        if self.event_study is not None:
            print("\n" + "-"*40)
            print("Event Study Estimates")
            print("-"*40)
            print(self.event_study.to_string(index=False))
        
        if self.pooled is not None:
            print("\n" + "-"*40)
            print("Pooled Estimates")
            print("-"*40)
            print(self.pooled.to_string(index=False))


def parse_formula(formula: str) -> Tuple[List[str], List[str]]:
    """
    Parse formula string like "~ controls | FE"
    
    Returns
    -------
    controls : list
        Control variables
    fixed_effects : list
        Fixed effects
    """
    if not formula or formula.strip() == "":
        return [], []
    
    # Remove leading ~ if present
    formula = formula.strip()
    if formula.startswith("~"):
        formula = formula[1:].strip()
    
    # Split by | (only 2 parts now)
    parts = [p.strip() for p in formula.split("|")]
    
    # Parse controls
    controls = []
    if len(parts) > 0 and parts[0]:
        controls = [c.strip() for c in parts[0].split("+") if c.strip()]
    
    # Parse fixed effects
    fixed_effects = []
    if len(parts) > 1 and parts[1]:
        fixed_effects = [fe.strip() for fe in parts[1].split("+") if fe.strip()]
    
    # Warn if there are too many parts (likely IV specification)
    if len(parts) > 2:
        warnings.warn("IV specification detected in formula but IV features are not supported. "
                     "Only controls and fixed effects will be used.")
    
    return controls, fixed_effects


def parse_interactions(interactions: str) -> List[str]:
    """
    Parse interaction specification like "~ var1 + var2"
    
    Returns
    -------
    interact_vars : list
        Variables to interact with treatment
    """
    if not interactions or interactions.strip() == "":
        return []
    
    # Remove leading ~ if present
    interactions = interactions.strip()
    if interactions.startswith("~"):
        interactions = interactions[1:].strip()
    
    # Parse interaction variables
    interact_vars = [v.strip() for v in interactions.split("+") if v.strip()]
    return interact_vars


def parse_cluster_formula(formula: str) -> List[str]:
    """
    Parse cluster formula like "~ cluster1 + cluster2"
    
    Returns
    -------
    clusters : list
        Clustering variables
    """
    if not formula or formula.strip() == "":
        return []
    
    # Remove leading ~ if present
    formula = formula.strip()
    if formula.startswith("~"):
        formula = formula[1:].strip()
    
    # Parse clusters
    clusters = [c.strip() for c in formula.split("+") if c.strip()]
    return clusters


class LPDiD:
    """
    Implements the Local Projections Difference-in-Differences (LP-DiD) estimator
    from Dube, Girardi, Jordà, and Taylor (2023).

    This class provides a flexible interface to estimate event-study-style
    treatment effects using a panel dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The panel dataset for analysis. Must be in long format.
    depvar : str
        The name of the dependent variable column.
    unit : str
        The name of the unit (e.g., individual, firm) identifier column.
    time : str
        The name of the time period identifier column.
    treat : str
        The name of the binary treatment indicator column (0 or 1).
    pre_window : int, optional
        The number of pre-treatment periods to estimate effects for. Must be >= 2.
        Defaults to 2 if not specified.
    post_window : int, optional
        The number of post-treatment periods to estimate effects for. Must be >= 0.
        Defaults to 0 if not specified.
    formula : str, optional
        A formula string to specify control variables and fixed effects,
        e.g., "~ x1 + x2 | fe1 + fe2".
    interactions : str, optional
        A formula-like string to specify variables that interact with the treatment
        indicator, e.g., "~ group_var". This allows for estimating heterogeneous effects.
    cluster_formula : str, optional
        A formula-like string to specify clustering variables, e.g., "~ cluster_var".
        Defaults to clustering by the `unit` identifier.
    ylags : int, optional
        The number of lags of the dependent variable to include as controls.
    dylags : int, optional
        The number of lags of the first-differenced dependent variable to include as controls.
    nevertreated : bool, default False
        If True, uses only never-treated units as the control group.
    min_time_selection : str, optional
        A pandas eval() expression to filter regression samples based on conditions at 
        the earlier period of each long-difference. This ensures comparability by selecting
        units based on their characteristics at the baseline period. The selection period is:
        - For post-treatment horizons (t-1 to t+h): conditions evaluated at t-1
        - For pre-treatment horizons (t-h to t-1): conditions evaluated at t-h
        Example: "employed == 1" to include only units that were employed at baseline.
    wildbootstrap : int, optional
        The number of iterations for wild bootstrap inference. If None, analytical
        standard errors are used.
    seed : int, optional
        A random seed for reproducibility, particularly for wild bootstrap.
    weights : str, optional
        The name of a column to be used as regression weights.
    n_jobs : int, default 1
        The number of CPU cores to use for parallel processing. -1 uses all available cores.
    lean : bool, default True
        If True, returns a more memory-efficient regression object from pyfixest. This reduces
        memory usage by not storing certain intermediate results.
    copy_data : bool, default False
        If True, creates a copy of the data before estimation in pyfixest. This can be useful
        to avoid modifying the original data but increases memory usage.
    mp_type : str, optional
        The multiprocessing start method to use ('fork', 'spawn', or 'forkserver'). 
        On Linux, 'spawn' is recommended to avoid Polars fork() warnings. 
        If None, uses the system default. Only affects parallel processing when n_jobs > 1.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 depvar: str,
                 unit: str,
                 time: str,
                 treat: str,
                 pre_window: Optional[int] = None,
                 post_window: Optional[int] = None,
                 formula: Optional[str] = None,
                 interactions: Optional[str] = None,
                 cluster_formula: Optional[str] = None,
                 ylags: Optional[int] = None,
                 dylags: Optional[int] = None,
                 nonabsorbing: Optional[Tuple[int, bool, bool]] = None,
                 nevertreated: bool = False,
                 nocomp: bool = False,
                 rw: bool = False,
                 min_time_controls: bool = False,
                 min_time_selection: Optional[str] = None,
                 swap_pre_diff: bool = False,
                 wildbootstrap: Optional[int] = None,
                 seed: Optional[int] = None,
                 weights: Optional[str] = None,
                 n_jobs: int = 1,
                 lean: bool = True,
                 copy_data: bool = False,
                 mp_type: Optional[str] = None,
                 # Backward compatibility parameters
                 controls: Optional[List[str]] = None,
                 absorb: Optional[List[str]] = None,
                 cluster: Optional[str] = None,
                 bootstrap: Optional[int] = None):
        
        # Store parameters
        self.data = data.copy()
        self.depvar = depvar
        self.unit = unit
        self.time = time
        self.treat = treat
        self.pre_window = pre_window
        self.post_window = post_window
        
        # Handle backward compatibility
        if controls is not None or absorb is not None:
            warnings.warn("Using deprecated parameters 'controls' and 'absorb'. "
                         "Please use the formula parameter instead: "
                         "formula='~ control1 + control2 | fe1 + fe2'", 
                         DeprecationWarning)
            self.controls = controls or []
            self.absorb = absorb or []
        elif formula:
            self.controls, self.absorb = parse_formula(formula)
        else:
            self.controls = []
            self.absorb = []
        
        # Parse interactions
        if interactions:
            self.interact_vars = parse_interactions(interactions)
        else:
            self.interact_vars = []
        
        # Handle cluster backward compatibility
        if cluster is not None:
            warnings.warn("Using deprecated parameter 'cluster'. "
                         "Please use cluster_formula instead: "
                         "cluster_formula='~ cluster1 + cluster2'",
                         DeprecationWarning)
            self.cluster_vars = [cluster]
        elif cluster_formula:
            self.cluster_vars = parse_cluster_formula(cluster_formula)
        else:
            self.cluster_vars = [unit]  # Default to unit clustering
        
        # Handle bootstrap backward compatibility
        if bootstrap is not None:
            warnings.warn("Using deprecated parameter 'bootstrap'. "
                         "Please use 'wildbootstrap' instead.",
                         DeprecationWarning)
            self.wildbootstrap = bootstrap
        else:
            self.wildbootstrap = wildbootstrap
            
        self.ylags = ylags
        self.dylags = dylags
        self.nonabsorbing = nonabsorbing
        self.nevertreated = nevertreated
        self.nocomp = nocomp
        self.rw = rw
        self.min_time_controls = min_time_controls
        self.min_time_selection = min_time_selection
        self.swap_pre_diff = swap_pre_diff
        self.seed = seed
        self.weights = weights
        self.n_jobs = n_jobs
        self.lean = lean
        self.copy_data = copy_data
        self.mp_type = mp_type
        
        # Set multiprocessing start method if specified
        self._configure_multiprocessing()
        
        # Validate inputs
        self._validate_inputs()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Warn about wild bootstrap limitations
        if self.wildbootstrap:
            warnings.warn(
                "Wild bootstrap inference only adjusts p-values and t-statistics. "
                "Standard errors and confidence intervals are still based on analytical inference. "
                "The wild bootstrap is used to provide more robust p-values under potential heteroskedasticity.",
                UserWarning
            )
        
        # Print initialization info
        self._print_init_info()
        
        # Prepare data
        print("Preparing data...")
        self._prepare_data()
        print(f"Data preparation complete. Dataset has {len(self.data)} observations.")
        
    def _configure_multiprocessing(self):
        """Configure warnings for multiprocessing (Ray handles process management automatically)"""
        # Ray automatically uses spawn-like behavior, so we don't need to configure multiprocessing
        # The mp_type parameter is now deprecated but kept for backward compatibility
        if self.mp_type is not None:
            warnings.warn(
                "The 'mp_type' parameter is deprecated when using Ray for parallelization. "
                "Ray automatically handles process management with proper memory isolation.",
                DeprecationWarning
            )

    def _validate_inputs(self):
        """Validate input parameters"""
        # Check if pre_window or post_window is specified
        if self.pre_window is None and self.post_window is None:
            raise ValueError("Please specify either post_window, pre_window, or both.")
        
        # Set defaults
        if self.pre_window is None:
            self.pre_window = 2
        if self.post_window is None:
            self.post_window = 0
            
        # Validate windows
        if self.pre_window < 2:
            raise ValueError("pre_window must be >= 2")
        if self.post_window < 0:
            raise ValueError("post_window must be >= 0")
        
        # Check binary treatment
        treat_vals = self.data[self.treat].dropna().unique()
        if not set(treat_vals).issubset({0, 1}):
            raise ValueError("Treatment variable must be binary (0/1)")
        
        # Parse nonabsorbing
        if self.nonabsorbing:
            L, notyet, firsttreat = self.nonabsorbing
            if not isinstance(L, int) or L < 1:
                raise ValueError("L in nonabsorbing must be a positive integer")
            if notyet and self.nevertreated:
                raise ValueError("Cannot specify both notyet and nevertreated")
    
    def _print_init_info(self):
        """Print initialization information including parallel processing details"""
        print("\n" + "="*60)
        print("LP-DiD Initialization (Linear Model)")
        print("="*60)
        
        # Basic info
        print(f"\nModel Information:")
        print(f"  Model type: Linear regression (OLS)")
        print(f"  Dependent variable: {self.depvar}")
        print(f"  Unit identifier: {self.unit}")
        print(f"  Time identifier: {self.time}")
        print(f"  Treatment indicator: {self.treat}")
        print(f"  Pre-treatment window: {self.pre_window}")
        print(f"  Post-treatment window: {self.post_window}")
        
        # Parallel processing info
        if self.n_jobs == -1:
            actual_cores = multiprocessing.cpu_count()
        else:
            actual_cores = min(self.n_jobs, multiprocessing.cpu_count())
        
        total_regressions = (self.pre_window - 1) + (self.post_window + 1)
        
        print(f"\nParallel Processing Configuration:")
        print(f"  System: {platform.system()} {platform.machine()}")
        print(f"  Available CPU cores: {multiprocessing.cpu_count()}")
        print(f"  Cores to be used: {actual_cores}")
        print(f"  Total regressions to run: {total_regressions}")
        
        # Show multiprocessing configuration
        current_mp_method = multiprocessing.get_start_method()
        print(f"  Multiprocessing start method: {current_mp_method}")
        if self.mp_type and self.mp_type != current_mp_method:
            print(f"  Requested mp_type: {self.mp_type} (differs from current method)")
        
        # Additional options
        if self.controls:
            print(f"\nControl variables: {', '.join(self.controls)}")
        if self.absorb:
            print(f"Fixed effects: {', '.join(self.absorb)}")
        if self.cluster_vars:
            print(f"Clustering variables: {', '.join(self.cluster_vars)}")
        if self.wildbootstrap:
            print(f"Wild bootstrap iterations: {self.wildbootstrap}")
        if self.nevertreated:
            print("Using only never-treated units as controls")
        
        print("="*60 + "\n")
    
    
    def _prepare_data(self):
        """Prepare data for estimation"""
        # Sort data
        self.data = self.data.sort_values([self.unit, self.time])
        
        # Set as panel
        self.data = self.data.set_index([self.unit, self.time])
        
        # Create first difference of treatment
        self.data['D_treat'] = self.data.groupby(level=0)[self.treat].diff()
        
        # Create group-specific treatment variables if interactions specified
        if self.interact_vars:
            for var in self.interact_vars:
                clean_var = var.replace("i.", "") if var.startswith("i.") else var
                
                # Get unique values of the interaction variable
                unique_vals = sorted(self.data[clean_var].dropna().unique())
                
                # Create separate treatment indicators for each group
                for val in unique_vals:
                    group_name = f"{clean_var}_{val}"
                    interaction_col = f"D_treat_{group_name}"
                    self.data[interaction_col] = (
                        self.data['D_treat'] * (self.data[clean_var] == val).astype(int)
                    )
        
        # Create lags if needed (using vectorized approach)
        if self.ylags:
            for lag in range(1, self.ylags + 1):
                self.data[f'L{lag}_{self.depvar}'] = (
                    self.data.groupby(level=0)[self.depvar].shift(lag)
                )
                self.controls.append(f'L{lag}_{self.depvar}')
        
        if self.dylags:
            # First create the first difference
            self.data[f'D_{self.depvar}'] = self.data.groupby(level=0)[self.depvar].diff()
            
            # Create differenced lags (using vectorized approach)
            for lag in range(1, self.dylags + 1):
                self.data[f'L{lag}_D_{self.depvar}'] = (
                    self.data.groupby(level=0)[f'D_{self.depvar}'].shift(lag)
                )
                self.controls.append(f'L{lag}_D_{self.depvar}')
        
        # Reset index for easier manipulation
        self.data = self.data.reset_index()

    def _identify_clean_controls(self):
        """Prepare data for clean control sample identification"""
        # Handle never-treated option
        if self.nevertreated:
            # Only never treated units
            max_treat = self.data.groupby(self.unit)[self.treat].max()
            never_treated = (max_treat == 0).astype(int)
            never_treated_df = never_treated.reset_index()
            never_treated_df.columns = [self.unit, 'never_treated']
            self.data = self.data.merge(never_treated_df, on=self.unit)
        
        # The actual CCS identification will be done in _generate_long_differences
        # after creating long_diff_data, based on treatment patterns and horizons

    
    def _generate_long_differences(self):
        """Generate long differences for dependent variable and apply CCS filtering"""
        # Determine processing strategy
        total_horizons = (self.post_window + 1) + (self.pre_window - 1)
        
        print(f"Generating long differences for {total_horizons} horizons...")
        print("Using optimized vectorized approach with integrated CCS filtering")
        
        # Always use vectorized approach
        self._generate_differences_vectorized_corrected()
    
    def _generate_differences_vectorized_corrected(self):
        """Corrected vectorized long difference generation to match pyfixest exactly"""
        print("Using corrected NumPy vectorized approach for long differences...")
        
        # Store results for each horizon
        all_results = []
        
        # Track min_time feature usage
        min_time_selection_applied = False
        min_time_controls_applied = False
        
        # Process POST-treatment horizons (0 to post_window)
        for h in range(self.post_window + 1):
            # Create long difference: y_{t+h} - y_{t-1}
            # Using shift(-h) for forward shift
            self.data[f'Dy_h{h}'] = (
                self.data.groupby(self.unit)[self.depvar].shift(-h) - 
                self.data.groupby(self.unit)[self.depvar].shift(1)
            )
            
            # Sample restriction for post-treatment
            # Include: newly treated (D_treat == 1) OR clean controls (treat at t+h == 0)
            clean_control_mask = self.data.groupby(self.unit)[self.treat].shift(-h) == 0
            sample_mask = (self.data['D_treat'] == 1) | clean_control_mask
            
            # Apply never-treated restriction if specified
            if self.nevertreated and 'never_treated' in self.data.columns:
                # For controls, also require never_treated = 1
                sample_mask = sample_mask & (
                    (self.data['D_treat'] == 1) | 
                    ((self.data['D_treat'] == 0) & (self.data['never_treated'] == 1))
                )
            
            # Apply min_time_selection if specified
            if self.min_time_selection:
                # For post-treatment (h >= 0): the earlier period is t-1
                # We need to evaluate the condition based on values at t-1
                try:
                    # Create a temporary mask for the selection condition
                    # We shift all relevant columns to get their t-1 values
                    selection_data = self.data.copy()
                    
                    # Shift all columns that might be used in the condition
                    for col in self.data.columns:
                        if col not in [self.unit, self.time, f'Dy_h{h}', 'D_treat']:
                            selection_data[f'{col}_at_selection'] = self.data.groupby(self.unit)[col].shift(1)
                    
                    # Modify the condition to use shifted column names
                    import re
                    modified_condition = self.min_time_selection
                    for col in self.data.columns:
                        if col not in [self.unit, self.time, f'Dy_h{h}', 'D_treat'] and f'{col}_at_selection' in selection_data.columns:
                            # Use word boundaries to ensure we only replace complete variable names
                            pattern = r'\b' + re.escape(col) + r'\b'
                            replacement = f'{col}_at_selection'
                            modified_condition = re.sub(pattern, replacement, modified_condition)
                    
                    # Evaluate the modified condition
                    selection_mask = selection_data.eval(modified_condition)
                    sample_mask = sample_mask & selection_mask
                    min_time_selection_applied = True
                    
                except Exception as e:
                    warnings.warn(f"Failed to apply min_time_selection for horizon {h}: {e}")
            
            # Create horizon data
            horizon_data = self.data[sample_mask].copy()
            horizon_data['Dy'] = horizon_data[f'Dy_h{h}']
            horizon_data['h'] = h
            horizon_data['is_pre'] = 0
            
            # Always add cntr_ columns for controls and fixed effects (except time)
            control_cols = self.controls + [fe for fe in self.absorb if fe != self.time]
            
            for col in control_cols:
                if col in self.data.columns:
                    # Always add cntr_ columns (shifted if min_time_controls, original otherwise)
                    if self.min_time_controls:
                        # For post-treatment: use control values from t-1
                        horizon_data[f'cntr_{col}'] = self.data.groupby(self.unit)[col].shift(1)[sample_mask].values
                        min_time_controls_applied = True
                    else:
                        horizon_data[f'cntr_{col}'] = horizon_data[col]
            
            # Drop the temporary column
            self.data = self.data.drop(columns=[f'Dy_h{h}'])
            
            all_results.append(horizon_data)
        
        # Process PRE-treatment horizons (2 to pre_window)
        for h in range(2, self.pre_window + 1):
            # Create long difference: y_{t-h} - y_{t-1}
            # Using shift(h) for backward shift
            self.data[f'Dy_h{-h}'] = (
                self.data.groupby(self.unit)[self.depvar].shift(h) - 
                self.data.groupby(self.unit)[self.depvar].shift(1)
            )
            
            # Sample restriction for pre-treatment
            # Include: newly treated (D_treat == 1) OR clean controls (treat == 0)
            sample_mask = (self.data['D_treat'] == 1) | (self.data[self.treat] == 0)
            
            # Apply never-treated restriction if specified
            if self.nevertreated and 'never_treated' in self.data.columns:
                sample_mask = sample_mask & (
                    (self.data['D_treat'] == 1) | 
                    ((self.data['D_treat'] == 0) & (self.data['never_treated'] == 1))
                )
            
            # Apply min_time_selection if specified
            if self.min_time_selection:
                # For pre-treatment (h < 0): the earlier period is t-h
                # We need to evaluate the condition based on values at t-h
                try:
                    # Create a temporary mask for the selection condition
                    # We shift all relevant columns to get their t-h values
                    selection_data = self.data.copy()
                    
                    # Shift all columns that might be used in the condition
                    for col in self.data.columns:
                        if col not in [self.unit, self.time, f'Dy_h{-h}', 'D_treat']:
                            selection_data[f'{col}_at_selection'] = self.data.groupby(self.unit)[col].shift(h)
                    
                    # Modify the condition to use shifted column names
                    import re
                    modified_condition = self.min_time_selection
                    for col in self.data.columns:
                        if col not in [self.unit, self.time, f'Dy_h{-h}', 'D_treat'] and f'{col}_at_selection' in selection_data.columns:
                            # Use word boundaries to ensure we only replace complete variable names
                            pattern = r'\b' + re.escape(col) + r'\b'
                            replacement = f'{col}_at_selection'
                            modified_condition = re.sub(pattern, replacement, modified_condition)
                    
                    # Evaluate the modified condition
                    selection_mask = selection_data.eval(modified_condition)
                    sample_mask = sample_mask & selection_mask
                    min_time_selection_applied = True
                    
                except Exception as e:
                    warnings.warn(f"Failed to apply min_time_selection for horizon {-h}: {e}")
            
            # Create horizon data
            horizon_data = self.data[sample_mask].copy()
            horizon_data['Dy'] = horizon_data[f'Dy_h{-h}']
            horizon_data['h'] = -h  # Negative for pre-treatment
            horizon_data['is_pre'] = 1
            
            # Always add cntr_ columns for controls and fixed effects (except time)
            control_cols = self.controls + [fe for fe in self.absorb if fe != self.time]
            
            for col in control_cols:
                if col in self.data.columns:
                    # Always add cntr_ columns (shifted if min_time_controls, original otherwise)
                    if self.min_time_controls:
                        # For pre-treatment: use control values from t-h
                        # FIXED: Use shift(h) instead of shift(1)
                        horizon_data[f'cntr_{col}'] = self.data.groupby(self.unit)[col].shift(h)[sample_mask].values
                        min_time_controls_applied = True
                    else:
                        horizon_data[f'cntr_{col}'] = horizon_data[col]
            
            # Drop the temporary column
            self.data = self.data.drop(columns=[f'Dy_h{-h}'])
            
            all_results.append(horizon_data)
        
        # Combine all horizons
        self.long_diff_data = pd.concat(all_results, ignore_index=True)
        
        # Drop all temporary Dy_h* columns
        temp_cols = [col for col in self.long_diff_data.columns if col.startswith('Dy_h')]
        self.long_diff_data = self.long_diff_data.drop(columns=temp_cols)

        # Remove any rows with missing Dy values
        initial_count = len(self.long_diff_data)
        self.long_diff_data = self.long_diff_data.dropna(subset=['Dy'])
        dropped_count = initial_count - len(self.long_diff_data)
        if dropped_count > 0:
            print(f"Dropped {dropped_count:,} observations with missing Dy values")
        
        # Sort by unit, time, and horizon for consistency
        self.long_diff_data = self.long_diff_data.sort_values([self.unit, self.time, 'h'])
        
        # Keep only relevant columns
        columns_to_keep = [
            self.unit,
            self.time,
            'h',
            'D_treat',
            'Dy',
            'is_pre'
        ]
        
        # Add control variables and their cntr_ counterparts
        for col in self.controls:
            if col in self.long_diff_data.columns:
                columns_to_keep.append(col)
            if f'cntr_{col}' in self.long_diff_data.columns:
                columns_to_keep.append(f'cntr_{col}')
        
        # Add fixed effect variables (except time which is already included) and their cntr_ counterparts
        for col in self.absorb:
            if col != self.time and col in self.long_diff_data.columns:
                columns_to_keep.append(col)
            if col != self.time and f'cntr_{col}' in self.long_diff_data.columns:
                columns_to_keep.append(f'cntr_{col}')
        
        # Add clustering variables if they're not already included
        for col in self.cluster_vars:
            if col not in columns_to_keep and col in self.long_diff_data.columns:
                columns_to_keep.append(col)
        
        # Add weights column if specified
        if self.weights and self.weights in self.long_diff_data.columns:
            columns_to_keep.append(self.weights)
        
        # Add interaction variables and their D_treat columns if present
        if self.interact_vars:
            for var in self.interact_vars:
                clean_var = var.replace("i.", "") if var.startswith("i.") else var
                if clean_var in self.long_diff_data.columns and clean_var not in columns_to_keep:
                    columns_to_keep.append(clean_var)
                # Add group-specific treatment columns
                group_cols = [col for col in self.long_diff_data.columns if col.startswith(f'D_treat_{clean_var}_')]
                columns_to_keep.extend([col for col in group_cols if col not in columns_to_keep])
        
        # Keep only unique columns in the order they appear
        columns_to_keep = list(dict.fromkeys(columns_to_keep))
        
        # Filter to keep only relevant columns
        self.long_diff_data = self.long_diff_data[columns_to_keep]
        
        print(f"Long differences generation complete. Created {len(self.long_diff_data):,} observations")
    def _compute_weights(self):
        """Compute weights for reweighted estimation"""
        if self.rw:
            # Create reweighting variables (placeholder implementation)
            for h in range(self.post_window + 1):
                self.data[f'reweight_{h}'] = 1.0
            self.data['reweight_0'] = 1.0

    def _build_regression_formula(self, y_var: str, fe_vars: List[str]) -> str:
        """Build regression formula with potential interactions"""
        # Build control variables part - always use cntr_ versions
        if self.interact_vars:
            # With interactions, we use group-specific treatment variables
            controls = []
            
            # Add cntr_ versions of control variables
            for ctrl in self.controls:
                controls.append(f"cntr_{ctrl}")
            
            # Add group-specific treatment indicators
            for var in self.interact_vars:
                clean_var = var.replace("i.", "") if var.startswith("i.") else var
                unique_vals = sorted(self.data[clean_var].dropna().unique())
                
                for val in unique_vals:
                    group_name = f"{clean_var}_{val}"
                    controls.append(f"D_treat_{group_name}")
        else:
            # Standard case: use cntr_ versions of controls and add D_treat
            controls = [f"cntr_{ctrl}" for ctrl in self.controls] + ['D_treat']
        
        # Build fixed effects part - use cntr_ versions except for time
        fe_cntr = []
        for fe in fe_vars:
            if fe == self.time:
                fe_cntr.append(fe)  # Time variable stays as is
            else:
                fe_cntr.append(f"cntr_{fe}")  # Other fixed effects use cntr_ version
        
        # Build formula parts
        controls_str = " + ".join(controls) if controls else "1"
        fe_str = " + ".join(fe_cntr) if fe_cntr else ""
        
        if fe_str:
            formula = f"{y_var} ~ {controls_str} | {fe_str}"
        else:
            formula = f"{y_var} ~ {controls_str}"
        
        return formula

    def _run_single_regression(self, horizon, is_pre=False):
        """
        Run a single local projection regression for a specific horizon.

        This internal method is the core of the estimation. It filters the long-format
        data for the specific horizon, builds the regression formula, and runs the
        estimation using `pyfixest`.

        Parameters
        ----------
        horizon : int
            The event time horizon for which to run the regression.
        is_pre : bool, default False
            A flag indicating whether the horizon is in the pre-treatment period.

        Returns
        -------
        dict or None
            A dictionary containing the regression results (coefficient, se, etc.)
            for the specified horizon, or None if the regression fails.
        """
        # Determine horizon value for filtering
        horizon_value = -horizon if is_pre else horizon
        
        # Filter long format data for this horizon (already CCS filtered)
        reg_data = self.long_diff_data[self.long_diff_data['h'] == horizon_value].copy()
        
        if reg_data.shape[0] == 0:
            return None
        
        # Build formula using 'Dy' as the dependent variable
        # Ensure unique fixed effects (avoid duplicates)
        fe_vars = [self.time] + self.absorb
        fe_vars = list(dict.fromkeys(fe_vars))  # Remove duplicates while preserving order
        formula = self._build_regression_formula('Dy', fe_vars)
        
        # Set up clustering
        if len(self.cluster_vars) == 1:
            vcov = {'CRV1': self.cluster_vars[0]}
        elif len(self.cluster_vars) == 2:
            # Two-way clustering - use CRV1 with string format
            vcov = {'CRV1': ' + '.join(self.cluster_vars)}
        else:
            # For 3+ clustering variables, also use string format
            vcov = {'CRV1': ' + '.join(self.cluster_vars)}
        
        # Add weights if specified
        # Note: reweighting is not implemented in long format yet
        if self.weights:
            use_weights = self.weights
        else:
            use_weights = None
        
        # Run the fixed-effects regression using pyfixest.
        try:
            if use_weights is not None:
                fit = pf.feols(
                    formula,
                    data=reg_data,
                    weights=use_weights,
                    vcov=vcov,
                    lean=self.lean,
                    copy_data=self.copy_data
                )
            else:
                fit = pf.feols(
                    formula,
                    data=reg_data,
                    vcov=vcov,
                    lean=self.lean,
                    copy_data=self.copy_data
                )
            
            nobs = fit._N
            
            # Store results dictionary
            results = {
                'horizon': -horizon if is_pre else horizon,
                'obs': nobs
            }
            
            # Handle different cases: group-specific effects vs main effect
            if self.interact_vars:
                # With interactions: we have separate coefficients for each group
                # Extract coefficients for each group
                group_coefficients = {}
                first_group_found = False
                
                for var in self.interact_vars:
                    clean_var = var.replace("i.", "") if var.startswith("i.") else var
                    unique_vals = sorted(self.data[clean_var].dropna().unique())
                    
                    for val in unique_vals:
                        group_name = f"{clean_var}_{val}"
                        coef_name = f"D_treat_{group_name}"
                        
                        if coef_name in fit.coef().index:
                            group_coef = fit.coef().loc[coef_name]
                            
                            # Store group-specific results
                            results[f'{clean_var}_{val}_coef'] = group_coef
                            
                            # Extract standard errors and inference for this group
                            if hasattr(fit, 'se') and coef_name in fit.se().index:
                                results[f'{clean_var}_{val}_se'] = fit.se().loc[coef_name]
                            if hasattr(fit, 'tstat') and coef_name in fit.tstat().index:
                                results[f'{clean_var}_{val}_t'] = fit.tstat().loc[coef_name]
                            if hasattr(fit, 'pvalue') and coef_name in fit.pvalue().index:
                                results[f'{clean_var}_{val}_p'] = fit.pvalue().loc[coef_name]
                            if hasattr(fit, 'confint'):
                                try:
                                    ci = fit.confint()
                                    if coef_name in ci.index:
                                        ci_row = ci.loc[coef_name]
                                        if hasattr(ci_row, 'iloc'):
                                            results[f'{clean_var}_{val}_ci_low'] = ci_row.iloc[0]
                                            results[f'{clean_var}_{val}_ci_high'] = ci_row.iloc[1]
                                        else:
                                            results[f'{clean_var}_{val}_ci_low'] = ci_row[0]
                                            results[f'{clean_var}_{val}_ci_high'] = ci_row[1]
                                except Exception:
                                    pass  # Skip CI if extraction fails
                            
                            # Use first group for backward compatibility main columns
                            if not first_group_found:
                                results['coefficient'] = group_coef
                                if hasattr(fit, 'se') and coef_name in fit.se().index:
                                    results['se'] = fit.se().loc[coef_name]
                                if hasattr(fit, 'tstat') and coef_name in fit.tstat().index:
                                    results['t'] = fit.tstat().loc[coef_name]
                                if hasattr(fit, 'pvalue') and coef_name in fit.pvalue().index:
                                    results['p'] = fit.pvalue().loc[coef_name]
                                if hasattr(fit, 'confint'):
                                    try:
                                        ci = fit.confint()
                                        if coef_name in ci.index:
                                            ci_row = ci.loc[coef_name]
                                            if hasattr(ci_row, 'iloc'):
                                                results['ci_low'] = ci_row.iloc[0]
                                                results['ci_high'] = ci_row.iloc[1]
                                            else:
                                                results['ci_low'] = ci_row[0]
                                                results['ci_high'] = ci_row[1]
                                    except Exception:
                                        pass  # Skip CI if extraction fails
                                first_group_found = True
                
                # If no groups were found, set NaN values
                if not first_group_found:
                    results['coefficient'] = np.nan
                    results['se'] = np.nan
                    results['t'] = np.nan
                    results['p'] = np.nan
                    results['ci_low'] = np.nan
                    results['ci_high'] = np.nan
                
            else:
                # Without interactions: extract main D_treat coefficient only
                if 'D_treat' not in fit.coef().index:
                    warnings.warn(f"D_treat coefficient not found for horizon {horizon}")
                    return None
                    
                # Extract main treatment coefficient
                coef = fit.coef().loc['D_treat']
                results['coefficient'] = coef
                
                # Handle inference for main effect
                if self.wildbootstrap:
                    # Use pyfixest's built-in wildboottest method
                    # NOTE: Wild bootstrap only adjusts p-values and t-statistics,
                    # standard errors and confidence intervals remain analytical
                    try:
                        # Set seed for this specific test if provided
                        if self.seed is not None:
                            np.random.seed(self.seed + horizon)  # Add horizon for variation
                        
                        wb_result = fit.wildboottest(
                            param="D_treat",
                            reps=self.wildbootstrap
                        )
                        
                        # Extract results from the wildboottest output (pandas Series)
                        if isinstance(wb_result, pd.Series):
                            # Extract wild bootstrap p-value (this is the main adjustment)
                            if 'Pr(>|t|)' in wb_result.index:
                                results['p'] = wb_result['Pr(>|t|)']
                            # Extract wild bootstrap t-statistic
                            if 't value' in wb_result.index:
                                results['t'] = wb_result['t value']
                        
                        # Use analytical standard errors and confidence intervals
                        # NOTE: These are NOT adjusted by wild bootstrap
                        if hasattr(fit, 'se') and 'D_treat' in fit.se().index:
                            results['se'] = fit.se().loc['D_treat']
                        if hasattr(fit, 'confint'):
                            try:
                                ci = fit.confint()
                                if 'D_treat' in ci.index:
                                    ci_row = ci.loc['D_treat']
                                    if hasattr(ci_row, 'iloc'):
                                        results['ci_low'] = ci_row.iloc[0]
                                        results['ci_high'] = ci_row.iloc[1]
                                    else:
                                        results['ci_low'] = ci_row[0]
                                        results['ci_high'] = ci_row[1]
                            except Exception:
                                pass  # Skip CI if extraction fails
                            
                    except Exception as e:
                        warnings.warn(f"Wild bootstrap failed, using analytical inference: {e}")
                        # Fall back to analytical standard errors
                        if hasattr(fit, 'se') and 'D_treat' in fit.se().index:
                            results['se'] = fit.se().loc['D_treat']
                        if hasattr(fit, 'tstat') and 'D_treat' in fit.tstat().index:
                            results['t'] = fit.tstat().loc['D_treat']
                        if hasattr(fit, 'pvalue') and 'D_treat' in fit.pvalue().index:
                            results['p'] = fit.pvalue().loc['D_treat']
                        if hasattr(fit, 'confint'):
                            try:
                                ci = fit.confint()
                                if 'D_treat' in ci.index:
                                    ci_row = ci.loc['D_treat']
                                    if hasattr(ci_row, 'iloc'):
                                        results['ci_low'] = ci_row.iloc[0]
                                        results['ci_high'] = ci_row.iloc[1]
                                    else:
                                        results['ci_low'] = ci_row[0]
                                        results['ci_high'] = ci_row[1]
                            except Exception:
                                pass  # Skip CI if extraction fails
                else:
                    # Use analytical standard errors
                    if hasattr(fit, 'se') and 'D_treat' in fit.se().index:
                        results['se'] = fit.se().loc['D_treat']
                    if hasattr(fit, 'tstat') and 'D_treat' in fit.tstat().index:
                        results['t'] = fit.tstat().loc['D_treat']
                    if hasattr(fit, 'pvalue') and 'D_treat' in fit.pvalue().index:
                        results['p'] = fit.pvalue().loc['D_treat']
                    if hasattr(fit, 'confint'):
                        try:
                            ci = fit.confint()
                            if 'D_treat' in ci.index:
                                ci_row = ci.loc['D_treat']
                                if hasattr(ci_row, 'iloc'):
                                    results['ci_low'] = ci_row.iloc[0]
                                    results['ci_high'] = ci_row.iloc[1]
                                else:
                                    results['ci_low'] = ci_row[0]
                                    results['ci_high'] = ci_row[1]
                        except Exception:
                            pass  # Skip CI if extraction fails
            
            return results
            
        except Exception as e:
            warnings.warn(f"Regression failed for horizon {horizon}: {e}")
            return None

    def fit(self):
        """
        Fit the Local Projections Difference-in-Differences (LP-DiD) model.

        This method executes the full LP-DiD estimation pipeline:
        1. Identifies clean control units that are never treated (if not already done).
        2. Constructs long-differenced outcomes for each horizon (if not already done).
        3. Computes weights for the regression if specified (if not already done).
        4. Runs separate regressions for each pre- and post-treatment horizon.
        5. Collates the results into an `LPDiDResults` object.
        
        If `build()` has been called previously, steps 1-3 are skipped and the method
        proceeds directly to running regressions using the already prepared data.

        Returns
        -------
        LPDiDResults
            An object containing the estimated event-study coefficients and other relevant information.
        """
        print("\n" + "="*60)
        print("Starting LP-DiD Estimation (Linear Model)")
        print("="*60)
        
        # Check if data has already been built
        if hasattr(self, 'long_diff_data'):
            print("\nUsing previously built data. Skipping data preparation steps.")
        else:
            # Step 1: Identify units that are never treated to serve as clean controls.
            print("\nStep 1: Identifying clean control samples...")
            self._identify_clean_controls()
            print("Clean control identification complete.")
            
            # Step 2: Create the long-differenced outcome variable for each horizon.
            print("\nStep 2: Generating long differences...")
            self._generate_long_differences()
            print("Long differences generation complete.")
            
            # Step 3: Compute regression weights if a weighting variable is specified.
            if self.rw:
                print("\nStep 3: Computing regression weights...")
                self._compute_weights()
                print("Weight computation complete.")
        
        # Step 4: Run event-study regressions for each specified horizon.
        print(f"\nStep 4: Running {(self.pre_window - 1) + (self.post_window + 1)} regressions...")
        
        # Determine actual number of cores to use
        if self.n_jobs == -1:
            n_cores = multiprocessing.cpu_count()
        else:
            n_cores = min(self.n_jobs, multiprocessing.cpu_count())
        
        # Prepare regression tasks
        regression_tasks = []
        
        # Add pre-treatment periods (h=2 to pre_window)
        for h in range(2, self.pre_window + 1):
            regression_tasks.append((h, True))  # True indicates pre-treatment
        
        # Add post-treatment periods (h=0 to post_window)
        for h in range(self.post_window + 1):
            regression_tasks.append((h, False))  # False indicates post-treatment
        
        # Run regressions (parallel or sequential)
        if n_cores > 1 and len(regression_tasks) > 1:
            print(f"Running regressions in parallel using {n_cores} cores with Ray...")
            
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                # Configure Ray for memory-only operation with 800GB object store
                ray.init(
                    num_cpus=n_cores,
                    ignore_reinit_error=True,
                    object_store_memory=800 * 1024 * 1024 * 1024,  # 800GB in bytes
                    _system_config={
                        "object_spilling_config": json.dumps({
                            "type": "disabled"  # Disable spilling completely
                        }),
                        "max_io_workers": 0  # No IO workers needed since no spilling
                    }
                )
                print(f"Ray initialized with 800GB object store memory (no disk spilling).")
            
            try:
                # Put the long_diff_data in Ray's object store for shared memory access
                long_diff_data_ref = ray.put(self.long_diff_data)
                
                # Submit all regression tasks to Ray
                ray_futures = []
                for h, is_pre in regression_tasks:
                    future = _run_single_regression_ray.remote(
                        horizon=h,
                        is_pre=is_pre,
                        long_diff_data=long_diff_data_ref,  # Pass reference instead of data
                        depvar=self.depvar,
                        time=self.time,
                        controls=self.controls,
                        absorb=self.absorb,
                        cluster_vars=self.cluster_vars,
                        interact_vars=self.interact_vars,
                        wildbootstrap=self.wildbootstrap,
                        seed=self.seed,
                        weights=self.weights,
                        lean=self.lean,
                        copy_data=self.copy_data
                    )
                    ray_futures.append(future)
                
                # Wait for all tasks to complete and collect results
                results = ray.get(ray_futures)
                
                # Filter out None results
                event_study_results = [r for r in results if r is not None]
                
                print(f"✓ Ray parallel estimation complete. Successfully estimated {len(event_study_results)} horizons.")
                
            except Exception as e:
                print(f"\nRay parallel processing failed: {type(e).__name__}: {e}")
                print("Falling back to sequential processing...")
                
                # Fall back to sequential processing
                event_study_results = []
                for i, (h, is_pre) in enumerate(regression_tasks):
                    print(f"  Progress: {i+1}/{len(regression_tasks)} - Horizon {-h if is_pre else h}...", end='\r')
                    result = self._run_single_regression(h, is_pre)
                    if result:
                        event_study_results.append(result)
                
                print(f"\nSequential estimation complete. Successfully estimated {len(event_study_results)} horizons.")
            
        else:
            # Sequential execution
            print("Running regressions sequentially...")
            event_study_results = []
            
            for i, (h, is_pre) in enumerate(regression_tasks):
                print(f"  Progress: {i+1}/{len(regression_tasks)} - Horizon {-h if is_pre else h}...", end='\r')
                result = self._run_single_regression(h, is_pre)
                if result:
                    event_study_results.append(result)
            
            print(f"\nSequential estimation complete. Successfully estimated {len(event_study_results)} horizons.")
        
        # Step 5: Collate results into a structured DataFrame.
        if event_study_results:
            event_study_df = pd.DataFrame(event_study_results)
        else:
            # If no regressions were successful, create an empty DataFrame with the expected structure.
            event_study_df = pd.DataFrame(columns=['horizon', 'coefficient', 'se', 't', 'p', 'ci_low', 'ci_high', 'obs'])

        # Normalize the event-study plot by setting the coefficient for h=-1 to 0.
        # This is a standard convention to show treatment effects relative to the period just before treatment.
        # For this period, all other metrics (se, t, etc.) are also set to 0 for consistency.
        if -1 not in event_study_df['horizon'].values:
            # Dynamically create the row with all zeros to handle any extra columns from interactions.
            cols = event_study_df.columns
            if len(cols) == 0:
                # Fallback if there were no other results from the regressions.
                cols = ['horizon', 'coefficient', 'se', 't', 'p', 'ci_low', 'ci_high', 'obs']
            
            h_minus_1_row = {col: 0.0 for col in cols}
            h_minus_1_row['horizon'] = -1
            
            event_study_df = pd.concat([pd.DataFrame([h_minus_1_row]), event_study_df], ignore_index=True)

        # Sort results by horizon for chronological plotting and analysis.
        event_study_df = event_study_df.sort_values('horizon').reset_index(drop=True)
        
        # Step 6: Package the results into a dedicated LPDiDResults object for easy access and plotting.
        results = LPDiDResults(
            event_study=event_study_df,
            depvar=self.depvar,
            pre_window=self.pre_window,
            post_window=self.post_window,
            control_group="Controls",
            treated_group="Treated"
        )
        
        return results
    
    def build(self):
        """
        Build the data for LP-DiD estimation without running regressions.
        
        This method executes the data preparation steps of the LP-DiD pipeline:
        1. Identifies clean control units that are never treated.
        2. Constructs long-differenced outcomes for each horizon.
        3. Computes weights for the regression if specified.
        
        Unlike `fit()`, this method does not run any regressions. It is useful when you want to:
        - Inspect the generated long-difference data before estimation
        - Use the prepared data for custom analyses
        - Save computation time when iterating on data preparation options
        
        Returns
        -------
        self
            Returns the LPDiD instance for method chaining.
            
        Notes
        -----
        After calling `build()`, you can access the prepared data using:
        - `get_long_diff_data()`: Returns the long-format difference data
        - Direct access to `self.long_diff_data` attribute
        
        Examples
        --------
        >>> # Build data without running regressions
        >>> lpdid = LPDiD(data, depvar='y', unit='id', time='t', treat='treat')
        >>> lpdid.build()
        >>> 
        >>> # Access the prepared data
        >>> long_data = lpdid.get_long_diff_data()
        >>> print(f"Generated {len(long_data)} observations across {long_data['h'].nunique()} horizons")
        >>> 
        >>> # Later run regressions if desired
        >>> results = lpdid.fit()
        """
        print("\n" + "="*60)
        print("Building LP-DiD Data (without running regressions)")
        print("="*60)
        
        # Step 1: Identify units that are never treated to serve as clean controls.
        print("\nStep 1: Identifying clean control samples...")
        self._identify_clean_controls()
        print("Clean control identification complete.")
        
        # Step 2: Create the long-differenced outcome variable for each horizon.
        print("\nStep 2: Generating long differences...")
        self._generate_long_differences()
        print("Long differences generation complete.")
        
        # Step 3: Compute regression weights if a weighting variable is specified.
        if self.rw:
            print("\nStep 3: Computing regression weights...")
            self._compute_weights()
            print("Weight computation complete.")
        
        # Print summary
        print("\n" + "-"*40)
        print("Data Build Summary")
        print("-"*40)
        if hasattr(self, 'long_diff_data'):
            print(f"Total observations: {len(self.long_diff_data):,}")
            print(f"Unique units: {self.long_diff_data[self.unit].nunique():,}")
            print(f"Unique time periods: {self.long_diff_data[self.time].nunique():,}")
            print(f"Horizons included: {sorted(self.long_diff_data['h'].unique())}")
            print(f"Treatment changes: {self.long_diff_data['D_treat'].sum():,}")
        
        print("\nData preparation complete. Use get_long_diff_data() to access the prepared data.")
        print("="*60 + "\n")
        
        return self
    
    def get_long_diff_data(self):
        """
        Get the long-format difference data generated during the fit or build process.
        
        This method returns the internally generated long-format dataset that contains
        all the long differences stacked vertically. Each row represents a unit-time-horizon
        combination with the following standardized columns:
        
        - All original data columns (unit, time, treatment, controls, etc.)
        - 'Dy': The long-differenced outcome variable
        - 'h': The horizon (-pre_window to post_window, with negative values for pre-treatment)
        - 'is_pre': Binary indicator (1 for pre-treatment horizons, 0 for post-treatment)
        - 'CCS': Clean control sample indicator (1 if included in clean control sample)
        
        Returns
        -------
        pd.DataFrame or None
            The long-format difference data if available, None if fit() or build() hasn't been called yet.
            
        Notes
        -----
        This long format is useful for:
        - Custom analyses that require all horizons in a single dataset
        - Pooled regressions across multiple horizons
        - Visualization and data exploration
        - Integration with other statistical packages that expect long format data
        
        Examples
        --------
        >>> # After fitting the model
        >>> lpdid = LPDiD(data, depvar='y', unit='id', time='t', treat='treat')
        >>> results = lpdid.fit()
        >>> 
        >>> # Get the long format data
        >>> long_data = lpdid.get_long_diff_data()
        >>> 
        >>> # Or just build the data without fitting
        >>> lpdid2 = LPDiD(data, depvar='y', unit='id', time='t', treat='treat')
        >>> lpdid2.build()
        >>> long_data2 = lpdid2.get_long_diff_data()
        >>> 
        >>> # Example: Run a pooled regression
        >>> import statsmodels.formula.api as smf
        >>> pooled_model = smf.ols('Dy ~ D_treat + C(h)', data=long_data[long_data['CCS']==1]).fit()
        """
        if hasattr(self, 'long_diff_data'):
            return self.long_diff_data.copy()
        else:
            warnings.warn("Long difference data not available. Please run fit() or build() first.")
            return None


class LPDiDPois(LPDiD):
    """
    Local Projections Difference-in-Differences Estimator with Poisson Regression
    
    Same parameters as LPDiD but uses Poisson regression instead of OLS.
    Suitable for count data and binary outcomes.
    """
    
    def _print_init_info(self):
        """Print initialization information including parallel processing details"""
        print("\n" + "="*60)
        print("LP-DiD Initialization (Poisson Model)")
        print("="*60)
        
        # Basic info
        print(f"\nModel Information:")
        print(f"  Model type: Poisson regression")
        print(f"  Dependent variable: {self.depvar}")
        print(f"  Unit identifier: {self.unit}")
        print(f"  Time identifier: {self.time}")
        print(f"  Treatment indicator: {self.treat}")
        print(f"  Pre-treatment window: {self.pre_window}")
        print(f"  Post-treatment window: {self.post_window}")
        
        # Parallel processing info
        if self.n_jobs == -1:
            actual_cores = multiprocessing.cpu_count()
        else:
            actual_cores = min(self.n_jobs, multiprocessing.cpu_count())
        
        total_regressions = (self.pre_window - 1) + (self.post_window + 1)
        
        print(f"\nParallel Processing Configuration:")
        print(f"  System: {platform.system()} {platform.machine()}")
        print(f"  Available CPU cores: {multiprocessing.cpu_count()}")
        print(f"  Cores to be used: {actual_cores}")
        print(f"  Total regressions to run: {total_regressions}")
        
        # Show multiprocessing configuration
        current_mp_method = multiprocessing.get_start_method()
        print(f"  Multiprocessing start method: {current_mp_method}")
        if self.mp_type and self.mp_type != current_mp_method:
            print(f"  Requested mp_type: {self.mp_type} (differs from current method)")
        
        # Additional options
        if self.controls:
            print(f"\nControl variables: {', '.join(self.controls)}")
        if self.absorb:
            print(f"Fixed effects: {', '.join(self.absorb)}")
        if self.cluster_vars:
            print(f"Clustering variables: {', '.join(self.cluster_vars)}")
        if self.wildbootstrap:
            print(f"Wild bootstrap iterations: {self.wildbootstrap}")
        if self.nevertreated:
            print("Using only never-treated units as controls")
        
        print("="*60 + "\n")
    
    def fit(self):
        """
        Fit the Local Projections Difference-in-Differences (LP-DiD) model using Poisson regression.

        This method executes the full LP-DiD estimation pipeline:
        1. Identifies clean control units that are never treated (if not already done).
        2. Constructs long-differenced outcomes for each horizon (if not already done).
        3. Computes weights for the regression if specified (if not already done).
        4. Runs separate Poisson regressions for each pre- and post-treatment horizon.
        5. Collates the results into an `LPDiDResults` object.
        
        If `build()` has been called previously, steps 1-3 are skipped and the method
        proceeds directly to running regressions using the already prepared data.

        Returns
        -------
        LPDiDResults
            An object containing the estimated event-study coefficients and other relevant information.
        """
        print("\n" + "="*60)
        print("Starting LP-DiD Estimation (Poisson Model)")
        print("="*60)
        
        # Check if data has already been built
        if hasattr(self, 'long_diff_data'):
            print("\nUsing previously built data. Skipping data preparation steps.")
        else:
            # Step 1: Identify units that are never treated to serve as clean controls.
            print("\nStep 1: Identifying clean control samples...")
            self._identify_clean_controls()
            print("Clean control identification complete.")
            
            # Step 2: Create the long-differenced outcome variable for each horizon.
            print("\nStep 2: Generating long differences...")
            self._generate_long_differences()
            print("Long differences generation complete.")
            
            # Step 3: Compute regression weights if a weighting variable is specified.
            if self.rw:
                print("\nStep 3: Computing regression weights...")
                self._compute_weights()
                print("Weight computation complete.")
        
        # Step 4: Run event-study regressions for each specified horizon.
        print(f"\nStep 4: Running {(self.pre_window - 1) + (self.post_window + 1)} regressions...")
        
        # Determine actual number of cores to use
        if self.n_jobs == -1:
            n_cores = multiprocessing.cpu_count()
        else:
            n_cores = min(self.n_jobs, multiprocessing.cpu_count())
        
        # Prepare regression tasks
        regression_tasks = []
        
        # Add pre-treatment periods (h=2 to pre_window)
        for h in range(2, self.pre_window + 1):
            regression_tasks.append((h, True))  # True indicates pre-treatment
        
        # Add post-treatment periods (h=0 to post_window)
        for h in range(self.post_window + 1):
            regression_tasks.append((h, False))  # False indicates post-treatment
        
        # Run regressions (parallel or sequential)
        if n_cores > 1 and len(regression_tasks) > 1:
            print(f"Running regressions in parallel using {n_cores} cores with Ray...")
            
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                # Configure Ray for memory-only operation with 800GB object store
                ray.init(
                    num_cpus=n_cores,
                    ignore_reinit_error=True,
                    object_store_memory=800 * 1024 * 1024 * 1024,  # 800GB in bytes
                    _system_config={
                        "object_spilling_config": json.dumps({
                            "type": "disabled"  # Disable spilling completely
                        }),
                        "max_io_workers": 0  # No IO workers needed since no spilling
                    }
                )
                print(f"Ray initialized with 800GB object store memory (no disk spilling).")
            
            try:
                # Put the long_diff_data in Ray's object store for shared memory access
                long_diff_data_ref = ray.put(self.long_diff_data)
                
                # Submit all regression tasks to Ray
                ray_futures = []
                for h, is_pre in regression_tasks:
                    future = _run_single_regression_ray_poisson.remote(
                        horizon=h,
                        is_pre=is_pre,
                        long_diff_data=long_diff_data_ref,  # Pass reference instead of data
                        depvar=self.depvar,
                        time=self.time,
                        controls=self.controls,
                        absorb=self.absorb,
                        cluster_vars=self.cluster_vars,
                        interact_vars=self.interact_vars,
                        wildbootstrap=self.wildbootstrap,
                        seed=self.seed,
                        weights=self.weights,
                        lean=self.lean,
                        copy_data=self.copy_data
                    )
                    ray_futures.append(future)
                
                # Wait for all tasks to complete and collect results
                results = ray.get(ray_futures)
                
                # Filter out None results
                event_study_results = [r for r in results if r is not None]
                
                print(f"✓ Ray parallel estimation complete. Successfully estimated {len(event_study_results)} horizons.")
                
            except Exception as e:
                print(f"\nRay parallel processing failed: {type(e).__name__}: {e}")
                print("Falling back to sequential processing...")
                
                # Fall back to sequential processing
                event_study_results = []
                for i, (h, is_pre) in enumerate(regression_tasks):
                    print(f"  Progress: {i+1}/{len(regression_tasks)} - Horizon {-h if is_pre else h}...", end='\r')
                    result = self._run_single_regression(h, is_pre)
                    if result:
                        event_study_results.append(result)
                
                print(f"\nSequential estimation complete. Successfully estimated {len(event_study_results)} horizons.")
            
        else:
            # Sequential execution
            print("Running regressions sequentially...")
            event_study_results = []
            
            for i, (h, is_pre) in enumerate(regression_tasks):
                print(f"  Progress: {i+1}/{len(regression_tasks)} - Horizon {-h if is_pre else h}...", end='\r')
                result = self._run_single_regression(h, is_pre)
                if result:
                    event_study_results.append(result)
            
            print(f"\nSequential estimation complete. Successfully estimated {len(event_study_results)} horizons.")
        
        # Step 5: Collate results into a structured DataFrame.
        if event_study_results:
            event_study_df = pd.DataFrame(event_study_results)
        else:
            # If no regressions were successful, create an empty DataFrame with the expected structure.
            event_study_df = pd.DataFrame(columns=['horizon', 'coefficient', 'se', 't', 'p', 'ci_low', 'ci_high', 'obs'])

        # Normalize the event-study plot by setting the coefficient for h=-1 to 0.
        # This is a standard convention to show treatment effects relative to the period just before treatment.
        # For this period, all other metrics (se, t, etc.) are also set to 0 for consistency.
        if -1 not in event_study_df['horizon'].values:
            # Dynamically create the row with all zeros to handle any extra columns from interactions.
            cols = event_study_df.columns
            if len(cols) == 0:
                # Fallback if there were no other results from the regressions.
                cols = ['horizon', 'coefficient', 'se', 't', 'p', 'ci_low', 'ci_high', 'obs']
            
            h_minus_1_row = {col: 0.0 for col in cols}
            h_minus_1_row['horizon'] = -1
            
            event_study_df = pd.concat([pd.DataFrame([h_minus_1_row]), event_study_df], ignore_index=True)

        # Sort results by horizon for chronological plotting and analysis.
        event_study_df = event_study_df.sort_values('horizon').reset_index(drop=True)
        
        # Step 6: Package the results into a dedicated LPDiDResults object for easy access and plotting.
        results = LPDiDResults(
            event_study=event_study_df,
            depvar=self.depvar,
            pre_window=self.pre_window,
            post_window=self.post_window,
            control_group="Controls",
            treated_group="Treated"
        )
        
        return results
    
    def _run_single_regression(self, horizon, is_pre=False):
        """Run a single LP-DiD regression using Poisson regression"""
        # Determine horizon value for filtering
        horizon_value = -horizon if is_pre else horizon
        
        # Filter long format data for this horizon (already CCS filtered)
        reg_data = self.long_diff_data[self.long_diff_data['h'] == horizon_value].copy()
        
        if reg_data.shape[0] == 0:
            return None
        
        # For Poisson regression, we need non-negative outcomes
        # Check if we have negative values and warn
        if reg_data['Dy'].min() < 0:
            warnings.warn(f"Negative values detected in long-differenced outcome. "
                         "Consider using swap_pre_diff=True for pre-treatment periods "
                         "or use regular LPDiD for continuous outcomes.")
            # Convert negative values to 0 for Poisson
            reg_data['Dy'] = np.maximum(reg_data['Dy'], 0)
        
        # Build formula using 'Dy' as the dependent variable
        # Ensure unique fixed effects (avoid duplicates)
        fe_vars = [self.time] + self.absorb
        fe_vars = list(dict.fromkeys(fe_vars))  # Remove duplicates while preserving order
        formula = self._build_regression_formula('Dy', fe_vars)
        
        # Set up clustering
        if len(self.cluster_vars) == 1:
            vcov = {'CRV1': self.cluster_vars[0]}
        elif len(self.cluster_vars) == 2:
            # Two-way clustering - use CRV1 with string format
            vcov = {'CRV1': ' + '.join(self.cluster_vars)}
        else:
            # For 3+ clustering variables, also use string format
            vcov = {'CRV1': ' + '.join(self.cluster_vars)}
        
        # Add weights if specified
        # Note: reweighting is not implemented in long format yet
        if self.weights:
            use_weights = self.weights
        else:
            use_weights = None
        
        # Run Poisson regression
        try:
            if use_weights is not None:
                fit = pf.fepois(
                    formula, 
                    data=reg_data, 
                    weights=use_weights,
                    vcov=vcov,
                    lean=self.lean,
                    copy_data=self.copy_data
                )
            else:
                fit = pf.fepois(
                    formula, 
                    data=reg_data,
                    vcov=vcov,
                    lean=self.lean,
                    copy_data=self.copy_data
                )
            
            nobs = fit._N
            
            # Store results dictionary
            results = {
                'horizon': -horizon if is_pre else horizon,
                'obs': nobs
            }
            
            # Handle different cases: group-specific effects vs main effect
            if self.interact_vars:
                # With interactions: we have separate coefficients for each group
                # Extract coefficients for each group
                group_coefficients = {}
                first_group_found = False
                
                for var in self.interact_vars:
                    clean_var = var.replace("i.", "") if var.startswith("i.") else var
                    unique_vals = sorted(self.data[clean_var].dropna().unique())
                    
                    for val in unique_vals:
                        group_name = f"{clean_var}_{val}"
                        coef_name = f"D_treat_{group_name}"
                        
                        if coef_name in fit.coef().index:
                            group_coef = fit.coef().loc[coef_name]
                            
                            # Store group-specific results
                            results[f'{clean_var}_{val}_coef'] = group_coef
                            
                            # Extract standard errors and inference for this group
                            if hasattr(fit, 'se') and coef_name in fit.se().index:
                                results[f'{clean_var}_{val}_se'] = fit.se().loc[coef_name]
                            if hasattr(fit, 'tstat') and coef_name in fit.tstat().index:
                                results[f'{clean_var}_{val}_t'] = fit.tstat().loc[coef_name]
                            if hasattr(fit, 'pvalue') and coef_name in fit.pvalue().index:
                                results[f'{clean_var}_{val}_p'] = fit.pvalue().loc[coef_name]
                            if hasattr(fit, 'confint'):
                                try:
                                    ci = fit.confint()
                                    if coef_name in ci.index:
                                        ci_row = ci.loc[coef_name]
                                        if hasattr(ci_row, 'iloc'):
                                            results[f'{clean_var}_{val}_ci_low'] = ci_row.iloc[0]
                                            results[f'{clean_var}_{val}_ci_high'] = ci_row.iloc[1]
                                        else:
                                            results[f'{clean_var}_{val}_ci_low'] = ci_row[0]
                                            results[f'{clean_var}_{val}_ci_high'] = ci_row[1]
                                except Exception:
                                    pass  # Skip CI if extraction fails
                            
                            # Use first group for backward compatibility main columns
                            if not first_group_found:
                                results['coefficient'] = group_coef
                                if hasattr(fit, 'se') and coef_name in fit.se().index:
                                    results['se'] = fit.se().loc[coef_name]
                                if hasattr(fit, 'tstat') and coef_name in fit.tstat().index:
                                    results['t'] = fit.tstat().loc[coef_name]
                                if hasattr(fit, 'pvalue') and coef_name in fit.pvalue().index:
                                    results['p'] = fit.pvalue().loc[coef_name]
                                if hasattr(fit, 'confint'):
                                    try:
                                        ci = fit.confint()
                                        if coef_name in ci.index:
                                            ci_row = ci.loc[coef_name]
                                            if hasattr(ci_row, 'iloc'):
                                                results['ci_low'] = ci_row.iloc[0]
                                                results['ci_high'] = ci_row.iloc[1]
                                            else:
                                                results['ci_low'] = ci_row[0]
                                                results['ci_high'] = ci_row[1]
                                    except Exception:
                                        pass  # Skip CI if extraction fails
                                first_group_found = True
                
                # If no groups were found, set NaN values
                if not first_group_found:
                    results['coefficient'] = np.nan
                    results['se'] = np.nan
                    results['t'] = np.nan
                    results['p'] = np.nan
                    results['ci_low'] = np.nan
                    results['ci_high'] = np.nan
                
            else:
                # Without interactions: extract main D_treat coefficient only
                if 'D_treat' not in fit.coef().index:
                    warnings.warn(f"D_treat coefficient not found for horizon {horizon}")
                    return None
                    
                # Extract main treatment coefficient
                coef = fit.coef().loc['D_treat']
                results['coefficient'] = coef
                
                # Handle inference for main effect
                if self.wildbootstrap:
                    # Note: Wild bootstrap for Poisson may not be as well-supported
                    warnings.warn("Wild bootstrap with Poisson regression may have limitations. "
                                 "Consider using analytical inference.")
                
                # Use analytical standard errors
                if hasattr(fit, 'se') and 'D_treat' in fit.se().index:
                    results['se'] = fit.se().loc['D_treat']
                if hasattr(fit, 'tstat') and 'D_treat' in fit.tstat().index:
                    results['t'] = fit.tstat().loc['D_treat']
                if hasattr(fit, 'pvalue') and 'D_treat' in fit.pvalue().index:
                    results['p'] = fit.pvalue().loc['D_treat']
                if hasattr(fit, 'confint'):
                    try:
                        ci = fit.confint()
                        if 'D_treat' in ci.index:
                            ci_row = ci.loc['D_treat']
                            if hasattr(ci_row, 'iloc'):
                                results['ci_low'] = ci_row.iloc[0]
                                results['ci_high'] = ci_row.iloc[1]
                            else:
                                results['ci_low'] = ci_row[0]
                                results['ci_high'] = ci_row[1]
                    except Exception:
                        pass  # Skip CI if extraction fails
            
            return results
            
        except Exception as e:
            warnings.warn(f"Poisson regression failed for horizon {horizon}: {e}")
            return None
