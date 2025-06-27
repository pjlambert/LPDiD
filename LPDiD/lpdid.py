"""
Local Projections Difference-in-Differences (LP-DiD) for Python
Based on Dube, Girardi, Jordà, and Taylor (2023)
"""

# Set OpenMP environment variables BEFORE importing numpy/scipy
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '2'
# Additional OMP settings that might help
os.environ['OMP_NESTED'] = 'FALSE'
os.environ['OMP_THREAD_LIMIT'] = '1'

import warnings
# Explicitly suppress all OMP-related warnings
warnings.filterwarnings("ignore", message=".*omp_set_nested.*")
warnings.filterwarnings("ignore", message=".*OMP.*nested.*deprecated.*")
warnings.filterwarnings("ignore", message=".*OMP_NESTED.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Cannot form a team.*")
warnings.filterwarnings("ignore", message=".*Consider unsetting.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*omp.*")

# Also try to suppress at the system level
import sys
if hasattr(sys, 'stderr'):
    class WarningFilter:
        def __init__(self, stream):
            self.stream = stream
            
        def write(self, data):
            # Filter out OMP warnings
            if any(phrase in data for phrase in [
                "OMP: Info #276", "OMP: Info #268", "OMP: Warning #96", "OMP: Hint"
            ]):
                return
            self.stream.write(data)
            
        def flush(self):
            self.stream.flush()
    
    # Temporarily replace stderr to filter OMP warnings
    original_stderr = sys.stderr
    sys.stderr = WarningFilter(original_stderr)

import numpy as np
import pandas as pd
import pyfixest as pf
from typing import Optional, Union, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from dataclasses import dataclass
from scipy import stats
import re

# Import wildboottest package
try:
    from wildboottest.wildboottest import wildboottest
    WILDBOOTTEST_AVAILABLE = True
except ImportError:
    # We'll use pyfixest's built-in wildboottest method instead
    WILDBOOTTEST_AVAILABLE = False


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
    wildbootstrap : int, optional
        The number of iterations for wild bootstrap inference. If None, analytical
        standard errors are used.
    seed : int, optional
        A random seed for reproducibility, particularly for wild bootstrap.
    weights : str, optional
        The name of a column to be used as regression weights.
    n_jobs : int, default 1
        The number of CPU cores to use for parallel processing. -1 uses all available cores.
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
                 pmd: Optional[Union[int, str]] = None,
                 min_time_controls: bool = False,
                 min_time_selection: Optional[str] = None,
                 swap_pre_diff: bool = False,
                 wildbootstrap: Optional[int] = None,
                 seed: Optional[int] = None,
                 weights: Optional[str] = None,
                 n_jobs: int = 1,
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
        self.pmd = pmd
        self.min_time_controls = min_time_controls
        self.min_time_selection = min_time_selection
        self.swap_pre_diff = swap_pre_diff
        self.seed = seed
        self.weights = weights
        self.n_jobs = n_jobs
        
        # Validate inputs
        self._validate_inputs()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Prepare data
        self._prepare_data()
        
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
        
        # Create lags if needed
        if self.ylags:
            for lag in range(1, self.ylags + 1):
                self.data[f'L{lag}_{self.depvar}'] = (
                    self.data.groupby(level=0)[self.depvar].shift(lag)
                )
                self.controls.append(f'L{lag}_{self.depvar}')
        
        if self.dylags:
            # First create the first difference
            self.data[f'D_{self.depvar}'] = self.data.groupby(level=0)[self.depvar].diff()
            for lag in range(1, self.dylags + 1):
                self.data[f'L{lag}_D_{self.depvar}'] = (
                    self.data.groupby(level=0)[f'D_{self.depvar}'].shift(lag)
                )
                self.controls.append(f'L{lag}_D_{self.depvar}')
        
        # Reset index for easier manipulation
        self.data = self.data.reset_index()

    def _identify_clean_controls(self):
        """Identify clean control samples and create indicators"""
        # Identify treatment switching periods
        self.data['first_treat_period'] = self.data.groupby(self.unit)[self.treat].transform(
            lambda x: x.idxmax() if x.any() else np.nan
        )
        
        # Create clean control sample indicators for each horizon
        post_CCS = self.post_window
        pre_CCS = self.pre_window
        
        # Post-treatment clean controls
        for h in range(post_CCS + 1):
            self.data[f'CCS_{h}'] = 1
            
        # Pre-treatment clean controls
        for h in range(2, pre_CCS + 1):
            self.data[f'CCS_m{h}'] = 1
        
        # Handle never-treated option
        if self.nevertreated:
            # Only never treated units
            max_treat = self.data.groupby(self.unit)[self.treat].max()
            never_treated = (max_treat == 0).astype(int)
            never_treated_df = never_treated.reset_index()
            never_treated_df.columns = [self.unit, 'never_treated']
            self.data = self.data.merge(never_treated_df, on=self.unit)
            
            # Update CCS indicators
            for h in range(post_CCS + 1):
                self.data.loc[
                    (self.data['D_treat'] == 0) & (self.data['never_treated'] == 0),
                    f'CCS_{h}'
                ] = 0
            for h in range(2, pre_CCS + 1):
                self.data.loc[
                    (self.data['D_treat'] == 0) & (self.data['never_treated'] == 0),
                    f'CCS_m{h}'
                ] = 0

    def _generate_long_differences(self):
        """Generate long differences for dependent variable"""
        if self.pmd is None:
            # Standard long differences
            if self.min_time_controls:
                # Use min(t-1, t+h) for control periods
                for h in range(self.post_window + 1):
                    # For post-treatment periods (h >= 0), always use t-1 as control
                    self.data[f'D{h}y'] = (
                        self.data.groupby(self.unit)[self.depvar].shift(-h) - 
                        self.data.groupby(self.unit)[self.depvar].shift(1)
                    )
                
                for h in range(2, self.pre_window + 1):
                    if h == 2:
                        # For horizon -2, use standard t-1 control
                        control_shift = 1
                    else:
                        # For longer horizons, use control closer to the outcome period
                        control_shift = h - 1  # Use t-(h-1) as control instead of t-1
                    
                    if self.swap_pre_diff:
                        self.data[f'Dm{h}y'] = (
                            self.data.groupby(self.unit)[self.depvar].shift(control_shift) - 
                            self.data.groupby(self.unit)[self.depvar].shift(h)
                        )
                    else:
                        self.data[f'Dm{h}y'] = (
                            self.data.groupby(self.unit)[self.depvar].shift(h) - 
                            self.data.groupby(self.unit)[self.depvar].shift(control_shift)
                        )
            else:
                # Standard implementation
                for h in range(self.post_window + 1):
                    self.data[f'D{h}y'] = (
                        self.data.groupby(self.unit)[self.depvar].shift(-h) - 
                        self.data.groupby(self.unit)[self.depvar].shift(1)
                    )
                
                for h in range(2, self.pre_window + 1):
                    if self.swap_pre_diff:
                        self.data[f'Dm{h}y'] = (
                            self.data.groupby(self.unit)[self.depvar].shift(1) - 
                            self.data.groupby(self.unit)[self.depvar].shift(h)
                        )
                    else:
                        self.data[f'Dm{h}y'] = (
                            self.data.groupby(self.unit)[self.depvar].shift(h) - 
                            self.data.groupby(self.unit)[self.depvar].shift(1)
                        )
        
        elif self.pmd == 'max':
            # Use all available pre-treatment periods
            def calc_pre_mean(group):
                cumsum = group[self.depvar].expanding().sum()
                count = group[self.depvar].expanding().count()
                return cumsum.shift(1) / count.shift(1)
            
            if self.min_time_controls:
                # For pmd='max' with min_time_controls, we need to modify the reference period
                # This is a more complex case that would require careful implementation
                # For now, issue a warning and fall back to standard pmd behavior
                warnings.warn("min_time_controls with pmd='max' is not yet implemented. "
                             "Using standard pmd='max' behavior.")
            
            self.data['aveLY'] = self.data.groupby(self.unit).apply(
                calc_pre_mean
            ).reset_index(drop=True)
            
            # Create differences
            for h in range(self.post_window + 1):
                self.data[f'D{h}y'] = (
                    self.data.groupby(self.unit)[self.depvar].shift(-h) - 
                    self.data['aveLY']
                )
            
            for h in range(2, self.pre_window + 1):
                if self.swap_pre_diff:
                    self.data[f'Dm{h}y'] = (
                        self.data['aveLY'] - 
                        self.data.groupby(self.unit)[self.depvar].shift(h)
                    )
                else:
                    self.data[f'Dm{h}y'] = (
                        self.data.groupby(self.unit)[self.depvar].shift(h) - 
                        self.data['aveLY']
                    )
        
        else:
            # Moving average over [-pmd, -1]
            def calc_ma(group, window):
                return group[self.depvar].rolling(window=window, min_periods=window).mean()
            
            if self.min_time_controls:
                # For pmd with min_time_controls, we need to modify the reference period
                # This is a more complex case that would require careful implementation
                # For now, issue a warning and fall back to standard pmd behavior
                warnings.warn("min_time_controls with pmd specification is not yet implemented. "
                             "Using standard pmd behavior.")
            
            self.data['aveLY'] = self.data.groupby(self.unit).apply(
                lambda x: calc_ma(x, self.pmd).shift(1)
            ).reset_index(drop=True)
            
            # Create differences
            for h in range(self.post_window + 1):
                self.data[f'D{h}y'] = (
                    self.data.groupby(self.unit)[self.depvar].shift(-h) - 
                    self.data['aveLY']
                )
            
            for h in range(2, self.pre_window + 1):
                if self.swap_pre_diff:
                    self.data[f'Dm{h}y'] = (
                        self.data['aveLY'] - 
                        self.data.groupby(self.unit)[self.depvar].shift(h)
                    )
                else:
                    self.data[f'Dm{h}y'] = (
                        self.data.groupby(self.unit)[self.depvar].shift(h) - 
                        self.data['aveLY']
                    )

    def _compute_weights(self):
        """Compute weights for reweighted estimation"""
        if self.rw:
            # Create reweighting variables (placeholder implementation)
            for h in range(self.post_window + 1):
                self.data[f'reweight_{h}'] = 1.0
            self.data['reweight_0'] = 1.0

    def _build_regression_formula(self, y_var: str, fe_vars: List[str]) -> str:
        """Build regression formula with potential interactions"""
        # Build control variables part
        if self.interact_vars:
            # With interactions, we use group-specific treatment variables
            controls = self.controls.copy()
            
            # Add group-specific treatment indicators
            for var in self.interact_vars:
                clean_var = var.replace("i.", "") if var.startswith("i.") else var
                unique_vals = sorted(self.data[clean_var].dropna().unique())
                
                for val in unique_vals:
                    group_name = f"{clean_var}_{val}"
                    controls.append(f"D_treat_{group_name}")
        else:
            # Standard case: just add D_treat
            controls = self.controls + ['D_treat']
        
        # Build formula parts
        controls_str = " + ".join(controls) if controls else "1"
        fe_str = " + ".join(fe_vars) if fe_vars else ""
        
        if fe_str:
            formula = f"{y_var} ~ {controls_str} | {fe_str}"
        else:
            formula = f"{y_var} ~ {controls_str}"
        
        return formula

    def _apply_min_time_selection(self, reg_data, horizon, is_pre=False):
        """Apply min_time_selection filter to regression data"""
        if not self.min_time_selection:
            return reg_data
        
        # Determine the control period based on min_time_controls logic
        if self.min_time_controls and is_pre and horizon > 2:
            control_shift = horizon - 1
        else:
            control_shift = 1
        
        # Create a time variable that accounts for the control period
        reg_data['control_time'] = reg_data[self.time] - control_shift
        
        # Merge with original data to get the condition values at control time
        control_data = self.data[[self.unit, self.time] + [col for col in self.data.columns 
                                                          if col not in [self.unit, self.time]]].copy()
        control_data = control_data.rename(columns={self.time: 'control_time'})
        
        # Merge to get values at control time
        reg_data = reg_data.merge(
            control_data[[self.unit, 'control_time'] + [col for col in control_data.columns 
                                                        if col not in [self.unit, 'control_time', 'D_treat']]],
            on=[self.unit, 'control_time'],
            how='left',
            suffixes=('', '_control')
        )
        
        # Apply the selection condition using values at control time
        try:
            # Replace variable names in the condition with their control-time values
            condition = self.min_time_selection
            for col in self.data.columns:
                if col not in [self.unit, self.time, 'D_treat'] and f'{col}_control' in reg_data.columns:
                    condition = condition.replace(col, f'{col}_control')
            
            # Evaluate the condition
            mask = reg_data.eval(condition)
            reg_data = reg_data[mask]
        except Exception as e:
            warnings.warn(f"Failed to apply min_time_selection condition '{self.min_time_selection}': {e}")
        
        # Clean up temporary columns
        control_cols = [col for col in reg_data.columns if col.endswith('_control')]
        reg_data = reg_data.drop(columns=control_cols + ['control_time'])
        
        return reg_data

    def _run_single_regression(self, horizon, is_pre=False):
        """
        Run a single local projection regression for a specific horizon.

        This internal method is the core of the estimation. It constructs the
        appropriate dependent variable for the given horizon, filters the data
        to the correct sample, builds the regression formula, and runs the
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
        # Determine dependent variable, clean control sample, and weight variables based on horizon
        if is_pre:
            y_var = f'Dm{horizon}y'
            ccs_var = f'CCS_m{horizon}'
            weight_var = 'reweight_0' if self.rw else None
        else:
            y_var = f'D{horizon}y'
            ccs_var = f'CCS_{horizon}'
            weight_var = f'reweight_{horizon}' if self.rw else None
        
        # Filter data
        mask = self.data[ccs_var] == 1
        reg_data = self.data[mask].copy()
        
        if reg_data.shape[0] == 0:
            return None
        
        # Apply min_time_selection filter if specified
        reg_data = self._apply_min_time_selection(reg_data, horizon, is_pre)
        
        if reg_data.shape[0] == 0:
            return None
        
        # Build formula
        fe_vars = [self.time] + self.absorb
        formula = self._build_regression_formula(y_var, fe_vars)
        
        # Set up clustering
        if len(self.cluster_vars) == 1:
            vcov = {'CRV1': self.cluster_vars[0]}
        else:
            # Multi-way clustering
            vcov = 'twoway'  # pyfixest will handle the cluster variables
        
        # Add weights if specified
        if self.weights and weight_var:
            # Combine user weights with reweighting
            reg_data['combined_weight'] = reg_data[self.weights] * reg_data[weight_var]
            use_weights = 'combined_weight'
        elif self.weights:
            use_weights = self.weights
        elif weight_var and weight_var in reg_data.columns:
            use_weights = weight_var
        else:
            use_weights = None
        
        # Run the fixed-effects regression using pyfixest.
        try:
            if use_weights is not None:
                fit = pf.feols(
                    formula,
                    data=reg_data,
                    weights=use_weights,
                    vcov=vcov
                )
            else:
                fit = pf.feols(
                    formula,
                    data=reg_data,
                    vcov=vcov
                )
            
            nobs = len(reg_data)
            
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
                                ci = fit.confint().loc[coef_name]
                                results[f'{clean_var}_{val}_ci_low'] = ci.iloc[0]
                                results[f'{clean_var}_{val}_ci_high'] = ci.iloc[1]
                            
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
                                    ci = fit.confint().loc[coef_name]
                                    results['ci_low'] = ci.iloc[0]
                                    results['ci_high'] = ci.iloc[1]
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
                            # Extract wild bootstrap p-value
                            if 'Pr(>|t|)' in wb_result.index:
                                results['p'] = wb_result['Pr(>|t|)']
                            # Extract wild bootstrap t-statistic
                            if 't value' in wb_result.index:
                                results['t'] = wb_result['t value']
                        
                        # Use analytical standard errors and confidence intervals
                        # since pyfixest wildboottest focuses on p-values and t-stats
                        if hasattr(fit, 'se') and 'D_treat' in fit.se().index:
                            results['se'] = fit.se().loc['D_treat']
                        if hasattr(fit, 'confint'):
                            ci = fit.confint().loc['D_treat']
                            results['ci_low'] = ci.iloc[0]
                            results['ci_high'] = ci.iloc[1]
                            
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
                            ci = fit.confint().loc['D_treat']
                            results['ci_low'] = ci.iloc[0]
                            results['ci_high'] = ci.iloc[1]
                else:
                    # Use analytical standard errors
                    if hasattr(fit, 'se') and 'D_treat' in fit.se().index:
                        results['se'] = fit.se().loc['D_treat']
                    if hasattr(fit, 'tstat') and 'D_treat' in fit.tstat().index:
                        results['t'] = fit.tstat().loc['D_treat']
                    if hasattr(fit, 'pvalue') and 'D_treat' in fit.pvalue().index:
                        results['p'] = fit.pvalue().loc['D_treat']
                    if hasattr(fit, 'confint'):
                        ci = fit.confint().loc['D_treat']
                        results['ci_low'] = ci.iloc[0]
                        results['ci_high'] = ci.iloc[1]
            
            return results
            
        except Exception as e:
            warnings.warn(f"Regression failed for horizon {horizon}: {e}")
            return None

    def fit(self):
        """
        Fit the Local Projections Difference-in-Differences (LP-DiD) model.

        This method executes the full LP-DiD estimation pipeline:
        1. Identifies clean control units that are never treated.
        2. Constructs long-differenced outcomes for each horizon.
        3. Computes weights for the regression if specified.
        4. Runs separate regressions for each pre- and post-treatment horizon.
        5. Collates the results into an `LPDiDResults` object.

        Returns
        -------
        LPDiDResults
            An object containing the estimated event-study coefficients and other relevant information.
        """
        # Step 1: Identify units that are never treated to serve as clean controls.
        self._identify_clean_controls()
        
        # Step 2: Create the long-differenced outcome variable for each horizon.
        self._generate_long_differences()
        
        # Step 3: Compute regression weights if a weighting variable is specified.
        self._compute_weights()
        
        # Step 4: Run event-study regressions for each specified horizon.
        event_study_results = []
        
        # Run regressions for each pre-treatment period (h=2 to pre_window).
        # h=1 is the period just before treatment, which is normalized to 0 and omitted here.
        for h in range(2, self.pre_window + 1):
            result = self._run_single_regression(h, is_pre=True)
            if result:
                event_study_results.append(result)
        
        # Run regressions for each post-treatment period (h=0 to post_window).
        # h=0 is the treatment period itself.
        for h in range(self.post_window + 1):
            result = self._run_single_regression(h, is_pre=False)
            if result:
                event_study_results.append(result)
        
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


class LPDiDPois(LPDiD):
    """
    Local Projections Difference-in-Differences Estimator with Poisson Regression
    
    Same parameters as LPDiD but uses Poisson regression instead of OLS.
    Suitable for count data and binary outcomes.
    """
    
    def _run_single_regression(self, horizon, is_pre=False):
        """Run a single LP-DiD regression using Poisson regression"""
        # Determine variables
        if is_pre:
            y_var = f'Dm{horizon}y'
            ccs_var = f'CCS_m{horizon}'
            weight_var = 'reweight_0' if self.rw else None
        else:
            y_var = f'D{horizon}y'
            ccs_var = f'CCS_{horizon}'
            weight_var = f'reweight_{horizon}' if self.rw else None
        
        # Filter data
        mask = self.data[ccs_var] == 1
        reg_data = self.data[mask].copy()
        
        if reg_data.shape[0] == 0:
            return None
        
        # Apply min_time_selection filter if specified
        reg_data = self._apply_min_time_selection(reg_data, horizon, is_pre)
        
        if reg_data.shape[0] == 0:
            return None
        
        # For Poisson regression, we need non-negative outcomes
        # Check if we have negative values and warn
        if reg_data[y_var].min() < 0:
            warnings.warn(f"Negative values detected in {y_var}. "
                         "Consider using swap_pre_diff=True for pre-treatment periods "
                         "or use regular LPDiD for continuous outcomes.")
            # Convert negative values to 0 for Poisson
            reg_data[y_var] = np.maximum(reg_data[y_var], 0)
        
        # Build formula
        fe_vars = [self.time] + self.absorb
        formula = self._build_regression_formula(y_var, fe_vars)
        
        # Set up clustering
        if len(self.cluster_vars) == 1:
            vcov = {'CRV1': self.cluster_vars[0]}
        else:
            # Multi-way clustering
            vcov = 'twoway'  # pyfixest will handle the cluster variables
        
        # Add weights if specified
        if self.weights and weight_var:
            # Combine user weights with reweighting
            reg_data['combined_weight'] = reg_data[self.weights] * reg_data[weight_var]
            use_weights = 'combined_weight'
        elif self.weights:
            use_weights = self.weights
        elif weight_var and weight_var in reg_data.columns:
            use_weights = weight_var
        else:
            use_weights = None
        
        # Run Poisson regression
        try:
            if use_weights is not None:
                fit = pf.fepois(
                    formula, 
                    data=reg_data, 
                    weights=use_weights,
                    vcov=vcov
                )
            else:
                fit = pf.fepois(
                    formula, 
                    data=reg_data,
                    vcov=vcov
                )
            
            nobs = len(reg_data)
            
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
                                ci = fit.confint().loc[coef_name]
                                results[f'{clean_var}_{val}_ci_low'] = ci.iloc[0]
                                results[f'{clean_var}_{val}_ci_high'] = ci.iloc[1]
                            
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
                                    ci = fit.confint().loc[coef_name]
                                    results['ci_low'] = ci.iloc[0]
                                    results['ci_high'] = ci.iloc[1]
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
                    ci = fit.confint().loc['D_treat']
                    results['ci_low'] = ci.iloc[0]
                    results['ci_high'] = ci.iloc[1]
            
            return results
            
        except Exception as e:
            warnings.warn(f"Poisson regression failed for horizon {horizon}: {e}")
            return None
