import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv(r"C:\Users\navya\OneDrive\Desktop\DM\Mobile_lifespan_app\Mobile_dataset.csv")
  # Ensure the correct file path

# 2. Drop unnecessary columns
df.drop(columns=["User ID"], inplace=True)

# 3. Handle missing values for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# 4. Encode categorical features
label_encoders = {}
categorical_cols = ["Device Model", "Operating System", "Gender"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 5. Estimate Lifespan (months) based on user behavior
df["Lifespan (months)"] = 36  # Default lifespan of 3 years (normal usage)

# Apply adjustments based on usage patterns
df.loc[df["Battery Drain (mAh/day)"] > 1800, "Lifespan (months)"] -= 12  # Heavy usage → reduce by 12 months
df.loc[df["Screen On Time (hours/day)"] > 5, "Lifespan (months)"] -= 6  # High screen time → reduce by 6 months
df.loc[df["User Behavior Class"] >= 4, "Lifespan (months)"] -= 6  # Power users → reduce by 6 months
df.loc[df["Battery Drain (mAh/day)"] < 800, "Lifespan (months)"] += 6  # Light users → increase by 6 months

# Ensure lifespan doesn't go below 12 months or above 60 months
df["Lifespan (months)"] = df["Lifespan (months)"].clip(12, 60)

# 6. Define Features and Target
features = ["App Usage Time (min/day)", "Screen On Time (hours/day)",
            "Battery Drain (mAh/day)", "Number of Apps Installed",
            "Data Usage (MB/day)", "Age", "Gender", "User Behavior Class"]

target = "Lifespan (months)"

X = df[features]
y = df[target]

# 7. Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 8. Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train a Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 10. Predict and evaluate
y_pred = model.predict(X_test)

# 11. Calculate Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calculate Accuracy using MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
accuracy = 100 - mape  # Accuracy in %

# 12. Print Results
print("\n📌 Mobile Phone Lifespan Prediction Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f} months")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} months")
print(f"R-Squared Score (R²): {r2:.2f}")
print(f"Model Accuracy: {accuracy:.2f}%")


# 13. Save the trained model for future use
with open("phone_lifespan_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n✅ Model trained and saved successfully as 'phone_lifespan_model.pkl'!")
