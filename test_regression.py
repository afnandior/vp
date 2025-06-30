import regression_module as reg

# Load your data
df = reg.load_data("your_data.xlsx")

# Call any regression model function
result = reg.run_ridge_regression(df, ["feature1", "feature2"], "target")

# Print metrics and file paths
print(result["metrics"])
print("Predictions saved at:", result["predictions_file"])
print("Plot saved at:", result["plot_file"])
