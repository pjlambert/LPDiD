import pandas as pd
from lpdid import LPDiD

# Adjust the import statement to ensure the `examples` folder is recognized
import sys
sys.path.append('/Users/pjl/Dropbox/lpdid/examples')
from example_simulation_data import generate_firm_simulation_data

def test_all_features():
    # Adjust the minimal dataset to ensure sufficient variation and observations
    data = pd.DataFrame({
        'entity_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'period': [1, 2, 1, 2, 1, 2, 1, 2],
        'treatment': [0, 1, 0, 0, 1, 1, 0, 1],
        'outcome': [10, 15, 20, 25, 30, 35, 40, 45],
        'x1': [1, 2, 3, 4, 5, 6, 7, 8],
        'x2': [6, 5, 4, 3, 2, 1, 8, 7],
        'cluster': [1, 1, 2, 2, 3, 3, 4, 4],
        'instrument': [0, 1, 0, 0, 1, 1, 0, 1],
    })

    # Basic LP-DiD estimation
    lpdid = LPDiD(
        data=data,
        depvar='outcome',
        unit='entity_id',
        time='period',
        treat='treatment',
        pre_window=2,
        post_window=1,
        formula="~ x1 + x2 | cluster",
        n_jobs=-1
    )
    results = lpdid.fit()
    print("Basic LP-DiD Results:")
    print(results.event_study)

    # Instrumental Variables
    lpdid_iv = LPDiD(
        data=data,
        depvar='outcome',
        unit='entity_id',
        time='period',
        treat='treatment',
        pre_window=2,
        post_window=1,
        formula="~ x1 + x2 | cluster | treatment ~ instrument",
        n_jobs=-1
    )
    results_iv = lpdid_iv.fit()
    print("IV Results:")
    print(results_iv.iv_diagnostics)

    # Treatment Effect Heterogeneity
    lpdid_het = LPDiD(
        data=data,
        depvar='outcome',
        unit='entity_id',
        time='period',
        treat='treatment',
        pre_window=2,
        post_window=1,
        formula="~ x1 + x2 | cluster",
        interactions="~ x1",
        n_jobs=-1
    )
    results_het = lpdid_het.fit()
    print("Heterogeneity Results:")
    print(results_het.event_study)

    # Wild Bootstrap Inference
    lpdid_boot = LPDiD(
        data=data,
        depvar='outcome',
        unit='entity_id',
        time='period',
        treat='treatment',
        pre_window=2,
        post_window=1,
        formula="~ x1 + x2 | cluster",
        wildbootstrap=999,
        n_jobs=-1
    )
    results_boot = lpdid_boot.fit()
    print("Wild Bootstrap Results:")
    print(results_boot.event_study)

    # Multi-way Clustering
    lpdid_multi = LPDiD(
        data=data,
        depvar='outcome',
        unit='entity_id',
        time='period',
        treat='treatment',
        pre_window=2,
        post_window=1,
        formula="~ x1 + x2 | cluster",
        cluster_formula="~ cluster",
        n_jobs=-1
    )
    results_multi = lpdid_multi.fit()
    print("Multi-way Clustering Results:")
    print(results_multi.event_study)

    # Weighted Estimation
    lpdid_weighted = LPDiD(
        data=data,
        depvar='outcome',
        unit='entity_id',
        time='period',
        treat='treatment',
        pre_window=2,
        post_window=1,
        formula="~ x1 + x2 | cluster",
        weights='x1',
        n_jobs=-1
    )
    results_weighted = lpdid_weighted.fit()
    print("Weighted Estimation Results:")
    print(results_weighted.event_study)

    # Non-absorbing Treatment
    lpdid_nonabs = LPDiD(
        data=data,
        depvar='outcome',
        unit='entity_id',
        time='period',
        treat='treatment',
        pre_window=2,
        post_window=1,
        formula="~ x1 + x2 | cluster",
        nonabsorbing=(1, False, False),
        n_jobs=-1
    )
    results_nonabs = lpdid_nonabs.fit()
    print("Non-absorbing Treatment Results:")
    print(results_nonabs.event_study)

    # Control Group Selection
    lpdid_control = LPDiD(
        data=data,
        depvar='outcome',
        unit='entity_id',
        time='period',
        treat='treatment',
        pre_window=2,
        post_window=1,
        formula="~ x1 + x2 | cluster",
        nevertreated=True,
        n_jobs=-1
    )
    results_control = lpdid_control.fit()
    print("Control Group Selection Results:")
    print(results_control.event_study)

    # Pre-mean Differencing
    lpdid_pmd = LPDiD(
        data=data,
        depvar='outcome',
        unit='entity_id',
        time='period',
        treat='treatment',
        pre_window=2,
        post_window=1,
        formula="~ x1 + x2 | cluster",
        pmd='max',
        n_jobs=-1
    )
    results_pmd = lpdid_pmd.fit()
    print("Pre-mean Differencing Results:")
    print(results_pmd.event_study)

    # Add a test case for basic functionality of LPDiD estimator using the simulated firm data

    # Generate the simulated firm data
    firm_data = generate_firm_simulation_data()

    # Basic LP-DiD estimation
    lpdid = LPDiD(
        data=firm_data,
        depvar='employment',
        unit='firm_id',
        time='time',
        treat='treated',
        pre_window=5,
        post_window=10,
        formula="~ bank_relationship",
        n_jobs=-1
    )
    results = lpdid.fit()
    print("Basic LP-DiD Results:")
    print(results.event_study)

if __name__ == "__main__":
    test_all_features()
    print("All features tested successfully.")