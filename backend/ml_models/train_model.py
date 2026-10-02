import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the dataset
print("Loading financial data...")
data = pd.read_csv("synthetic_financial_data.csv")

# 2. Define the features (inputs) and target (output)
X = data[["Monthly_Income", "Essential_Expenses", "Discretionary_Expenses", "Liquid_Savings", "Monthly_Debt"]]
y = data["Financial_Health_Class"]

# 3. Initialize and train the Random Forest
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# 4. Save the trained model so FastAPI can use it later
joblib.dump(rf_model, "financial_rf_model.joblib")
print("Model saved successfully as 'financial_rf_model.joblib'")

# 5. Extract Feature Importance for your midterm report
importance = dict(zip(X.columns, rf_model.feature_importances_))
print("\nFeature Importance (Which metric matters most):")
for feature, score in sorted(importance.items(), key=lambda item: item[1], reverse=True):
    print(f"- {feature}: {score:.4f}")