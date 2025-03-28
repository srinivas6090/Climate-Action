import numpy as np
import pandas as pd

def generate_synthetic_data():
    np.random.seed(42)

    years = np.arange(2000, 2021)
    co2_levels = 400 + (years - 2000) * 2 + np.random.normal(0, 5, len(years))  # Simulated CO2 levels
    temperature = 15 + (co2_levels - 400) * 0.02 + np.random.normal(0, 0.5, len(years))  # Simulated temperatures

    data = pd.DataFrame({
        'Year': years,
        'CO2 Levels': co2_levels,
        'Temperature': temperature
    })

    # Save the synthetic data to a CSV file
    data.to_csv('data/synthetic_climate_data.csv', index=False)
    print("Synthetic climate data generated and saved to data/synthetic_climate_data.csv")

if __name__ == "__main__":
    generate_synthetic_data()
