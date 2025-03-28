from generate_data import generate_synthetic_data
from climate_model import load_data, train_model, plot_results, simulate_climate_action

# Step 1: Generate synthetic data
generate_synthetic_data()

# Step 2: Load and train the model
data = load_data()
model = train_model(data)

# Step 3: Plot results
plot_results(data, model)

# Step 4: Simulate a climate action
co2_reduction = 50  # Example reduction in CO2 levels
predicted_temps = simulate_climate_action(model, co2_reduction)
print(f"Predicted Temperatures after {co2_reduction} ppm reduction in CO2 levels: {predicted_temps}")
