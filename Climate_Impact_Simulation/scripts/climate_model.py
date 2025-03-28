import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_data():
    return pd.read_csv('data/synthetic_climate_data.csv')

def train_model(data):
    X = data[['CO2 Levels']]  # Features
    y = data['Temperature']    # Target variable

    model = LinearRegression()
    model.fit(X, y)
    
    return model

def simulate_climate_action(model, co2_reduction):
    # Predict temperature change based on CO2 reduction
    co2_levels = np.array([400, 400 - co2_reduction]).reshape(-1, 1)  # CO2 before and after reduction
    predicted_temp = model.predict(co2_levels)
    return predicted_temp

def plot_results(data, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['CO2 Levels'], data['Temperature'], color='blue', label='Actual Data')
    X_range = np.linspace(380, 450, 100).reshape(-1, 1)
    plt.plot(X_range, model.predict(X_range), color='red', label='Model Prediction')
    
    plt.title('Climate Impact Model')
    plt.xlabel('CO2 Levels (ppm)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    data = load_data()
    model = train_model(data)
    plot_results(data, model)
    
    # Simulate a climate action by reducing CO2 levels
    co2_reduction = 50  # Example reduction in CO2 levels
    predicted_temps = simulate_climate_action(model, co2_reduction)
    print(f"Predicted Temperatures after {co2_reduction} ppm reduction in CO2 levels: {predicted_temps}")
