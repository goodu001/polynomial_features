import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
# This assumes the CSV file 'Braking_data.xlsx - Sheet1.csv' is available.
try:
    df = pd.read_csv('Braking_data.xlsx - Sheet1.csv')
except FileNotFoundError:
    print("Error: 'Braking_data.xlsx - Sheet1.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Define the independent and dependent variables
X = df[['Speed_kmph']]
y = df['Braking_Distance_m']

# 1. Create a polynomial feature transformer with degree 2
# This will transform the single 'Speed_kmph' column into two columns:
# one for Speed_kmph and one for (Speed_kmph)^2.
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# 2. Create and train a new linear regression model on the polynomial features
# We use LinearRegression because it's a linear model, but it's applied
# to the non-linear polynomial features.
model_poly = LinearRegression()
model_poly.fit(X_poly, y)

# 3. Measure the model's efficiency
# R-squared measures the proportion of the variance in the dependent variable
# that is predictable from the independent variable(s).
r_squared_poly = model_poly.score(X_poly, y)

# Root Mean Squared Error (RMSE) measures the average magnitude of the errors.
# A lower value indicates a better fit.
y_pred_poly = model_poly.predict(X_poly)
rmse_poly = np.sqrt(mean_squared_error(y, y_pred_poly))

# Print the results
print("--- Polynomial Regression Model (Degree 2) ---")
print(f"Intercept: {model_poly.intercept_}")
print(f"Coefficients: {model_poly.coef_}")
print(f"R-squared: {r_squared_poly:.4f}")
print(f"RMSE: {rmse_poly:.2f}")

# 4. Visualization: Plot the original data and the polynomial curve
plt.figure(figsize=(10, 6))

# Plot the original scatter data points
plt.scatter(X, y, color='blue', label='Actual Data')

# Sort the data by speed for a smooth curve plot
sorted_zip = sorted(zip(X['Speed_kmph'], y_pred_poly))
X_sorted, y_pred_sorted = zip(*sorted_zip)

# Plot the predicted polynomial curve
plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Polynomial Regression Curve')

# Add plot labels and title
plt.title('Polynomial Regression: Braking Distance vs. Speed')
plt.xlabel('Speed (km/h)')
plt.ylabel('Braking Distance (m)')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('polynomial_regression_plot.png')
plt.show()
