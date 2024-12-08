# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
target = "MedHouseVal"

# Splitting the data into features and target
X = df.drop(columns=[target])
y = df[target]

# Data preprocessing
# Handle missing values (if any)
X.fillna(X.mean(), inplace=True)

# Visualize feature distributions before scaling
X.describe().T[['mean', 'std', 'min', 'max']].plot(kind='bar', figsize=(10, 6), title='Feature Statistics Before Scaling')
plt.show()

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualize feature distributions after scaling
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
scaled_df.describe().T[['mean', 'std']].plot(kind='bar', figsize=(10, 6), title='Feature Statistics After Scaling')
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models Initialization
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=0.1)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Model Fitting
linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Model performance evaluation
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Evaluate all models
models = {
    "Linear Regression": y_pred_linear,
    "Ridge Regression": y_pred_ridge,
    "Lasso Regression": y_pred_lasso,
    "Random Forest Regression": y_pred_rf,
}

results = {model: evaluate_model(y_test, preds, model) for model, preds in models.items()}

# Visualize model performance
performance_df = pd.DataFrame(results, index=['MSE', 'R2']).T
performance_df.plot(kind='bar', figsize=(10, 6), title='Model Performance Comparison')
plt.ylabel('Score')
plt.show()

# Coefficient Analysis for Linear, Ridge, and Lasso
for model_name, model in zip(["Linear Regression", "Ridge Regression", "Lasso Regression"],
                             [linear_model, ridge_model, lasso_model]):
    coefficients = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
    coefficients.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Feature Coefficients ({model_name})')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.show()

# Random Forest Feature Importance
feature_importances = pd.Series(rf_model.feature_importances_, index=data.feature_names).sort_values(ascending=False)
feature_importances.plot(kind='bar', figsize=(10, 6))
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()

# Ridge and Lasso regularization effects
alphas = np.logspace(-3, 3, 50)
ridge_coefs, lasso_coefs = [], []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha)
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)

# Ridge regularization path
plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_coefs)
plt.xscale('log')
plt.title('Ridge Coefficients as a Function of Regularization (Alpha)')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Values')
plt.show()

# Lasso regularization path
plt.figure(figsize=(10, 6))
plt.plot(alphas, lasso_coefs)
plt.xscale('log')
plt.title('Lasso Coefficients as a Function of Regularization (Alpha)')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Values')
plt.show()

