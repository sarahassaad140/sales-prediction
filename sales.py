# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load Dataset
data = pd.read_csv('Advertising.csv')

# Step 2: Explore the Dataset
print("Dataset Head:\n", data.head())
print("\nDataset Info:")
data.info()
print("\nSummary Statistics:")
print(data.describe())

# Step 3: Check and Handle Missing Values
print("\nMissing Values:\n", data.isnull().sum())
data = data.dropna()  # Drop rows with missing values if any

# Step 4: Data Visualization
# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Step 5: Feature Selection and Preprocessing
X = data[['TV', 'Radio', 'Newspaper']]  # Features
y = data['Sales']  # Target variable

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)

# Print model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
for feature, coef in zip(['TV', 'Radio', 'Newspaper'], model.coef_):
    print(f"{feature} Coefficient: {coef}")

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Step 8: Cross-Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")

# Step 9: Save the Model
joblib.dump(model, 'sales_prediction_model.pkl')
print("\nModel saved as 'sales_prediction_model.pkl'.")

# Step 10: Visualization
# Actual vs Predicted Sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.show()

# Visualizing Feature Importance
feature_importance = pd.DataFrame({'Feature': ['TV', 'Radio', 'Newspaper'], 'Importance': model.coef_})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False))

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance")
plt.show()
