# regression_module.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Create outputs folder if not exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# ===============================
# 📌 1. Load data function
# ===============================
def load_data(file_path=None, manual_data=None, columns=None):
    """
    Load data from CSV, Excel, or manual input.
    """
    if file_path:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)
        else:
            raise Exception("Unsupported file type")
    elif manual_data and columns:
        df = pd.DataFrame(manual_data, columns=columns)
    else:
        raise Exception("Provide either file_path or manual_data + columns")
    
    return df

# ===============================
# 📌 2. Generic Regression Function Template
# ===============================
def regression_template(df, feature_columns, target_column, model, model_name):
    X = df[feature_columns]
    y = df[target_column]
    
    model.fit(X, y)
    y_pred = model.predict(X)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y, y_pred)

    # Save predictions
    pred_df = pd.DataFrame({
        "Actual": y,
        "Predicted": y_pred
    })
    predictions_file = f"outputs/{model_name}_predictions.csv"
    pred_df.to_csv(predictions_file, index=False)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(y, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} Predictions")
    plot_file = f"outputs/{model_name}_plot.png"
    plt.savefig(plot_file)
    plt.close()

    # Return results
    return {
        "model": model_name,
        "metrics": {
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        },
        "predictions_file": predictions_file,
        "plot_file": plot_file
    }

# ===============================
# 📌 3. Regression Models
# ===============================

def run_linear_regression(df, feature_columns, target_column):
    model = LinearRegression()
    return regression_template(df, feature_columns, target_column, model, "LinearRegression")

def run_polynomial_regression(df, feature_columns, target_column, degree=2):
    X = df[feature_columns]
    y = df[target_column]
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    mse = mean_squared_error(y, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y, y_pred)

    # Save predictions
    pred_df = pd.DataFrame({
        "Actual": y,
        "Predicted": y_pred
    })
    predictions_file = f"outputs/PolynomialRegression_predictions.csv"
    pred_df.to_csv(predictions_file, index=False)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(y, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Polynomial Regression Predictions")
    plot_file = f"outputs/PolynomialRegression_plot.png"
    plt.savefig(plot_file)
    plt.close()

    return {
        "model": "Polynomial Regression",
        "metrics": {
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        },
        "predictions_file": predictions_file,
        "plot_file": plot_file
    }

def run_ridge_regression(df, feature_columns, target_column, alpha=1.0):
    model = Ridge(alpha=alpha)
    return regression_template(df, feature_columns, target_column, model, "RidgeRegression")

def run_lasso_regression(df, feature_columns, target_column, alpha=1.0):
    model = Lasso(alpha=alpha)
    return regression_template(df, feature_columns, target_column, model, "LassoRegression")

def run_elasticnet_regression(df, feature_columns, target_column, alpha=1.0, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    return regression_template(df, feature_columns, target_column, model, "ElasticNetRegression")

def run_decision_tree_regression(df, feature_columns, target_column):
    model = DecisionTreeRegressor()
    return regression_template(df, feature_columns, target_column, model, "DecisionTreeRegression")

def run_random_forest_regression(df, feature_columns, target_column, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators)
    return regression_template(df, feature_columns, target_column, model, "RandomForestRegression")

def run_gradient_boosting_regression(df, feature_columns, target_column, n_estimators=100):
    model = GradientBoostingRegressor(n_estimators=n_estimators)
    return regression_template(df, feature_columns, target_column, model, "GradientBoostingRegression")

# ===============================
# 📌 4. Logistic Regression (classification)
# ===============================

def run_logistic_regression(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Accuracy
    accuracy = model.score(X, y)

    # Save predictions
    pred_df = pd.DataFrame({
        "Actual": y,
        "Predicted": y_pred
    })
    predictions_file = f"outputs/LogisticRegression_predictions.csv"
    pred_df.to_csv(predictions_file, index=False)

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(y, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Logistic Regression Predictions")
    plot_file = f"outputs/LogisticRegression_plot.png"
    plt.savefig(plot_file)
    plt.close()

    return {
        "model": "Logistic Regression",
        "metrics": {
            "Accuracy": accuracy
        },
        "predictions_file": predictions_file,
        "plot_file": plot_file
    }

# ===============================
# 📌 5. Example usage
# ===============================

if __name__ == "__main__":
    # Example usage when running as standalone script
    df = load_data("example_data.csv")
    result = run_linear_regression(df, ["feature1", "feature2"], "target")
    print(result)
