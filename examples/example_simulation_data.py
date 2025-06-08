import numpy as np
import pandas as pd

def generate_firm_simulation_data(seed=757):
    # Set parameters
    N_firms = 2000  # Increase the number of firms
    T_periods = 50  # Keep the number of time periods constant
    e_sd = 10
    alpha0 = 5
    alpha1 = 0.2

    np.random.seed(seed)

    # Create Base Data
    df = pd.DataFrame({
        'firm_id': np.repeat(np.arange(1, N_firms + 1), T_periods),
        'time': np.tile(np.arange(1, T_periods + 1), N_firms),
        'employment': np.random.normal(100, e_sd, N_firms * T_periods),
        'bank_relationship': np.random.choice(['A', 'B', 'C'], N_firms * T_periods, replace=True)
    })

    # Add more variation to the simulated data to reduce collinearity
    df['employment'] += np.random.normal(0, e_sd * 2, len(df))  # Add random noise to employment
    df['bank_relationship'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], len(df), replace=True)  # Increase categories

    # Assign treatment (bank relationship failure) staggered over time
    treatment_start = np.random.choice(np.arange(10, 40), N_firms, replace=True)
    df['treatment_start'] = np.repeat(treatment_start, T_periods)
    df['treated'] = np.where(df['time'] >= df['treatment_start'], 1, 0)

    # Bake in treatment effect
    df['rel_time'] = df['time'] - df['treatment_start']
    df['rel_time'] = np.where(df['treated'] == 1, df['rel_time'], -1)
    df['treatment_effect'] = np.where(
        df['rel_time'] >= 0,
        alpha0 + alpha1 * df['rel_time'],
        0
    )
    df['employment'] += df['treatment_effect']

    # Print basic properties of the data
    print("Data Properties:")
    print("Number of firms:", N_firms)
    print("Number of time periods:", T_periods)
    print("Total observations:", len(df))  # Print the total number of observations
    print("Sample data:")
    print(df.head())

    return df

# Example usage
if __name__ == "__main__":
    firm_data = generate_firm_simulation_data()
    print("Simulation complete.")