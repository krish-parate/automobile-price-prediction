import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load data
df = pd.read_csv("data/car_data.csv")

# Remove rows with Ask For Priced
df = df[df["Price"] != "Ask For Price"]

# Clean Price
df["Price"] = df["Price"].str.replace(",", "")
df["Price"] = df["Price"].astype(int)

# Clean kms_driven
df["kms_driven"] = df["kms_driven"].str.replace(" kms", "")
df["kms_driven"] = df["kms_driven"].str.replace(",", "")
df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors="coerce")

# Clean year
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Remove missing values
df = df.dropna()

# Features and target
X = df[["name", "company", "year", "kms_driven", "fuel_type"]]
y = df["Price"]

# Encode text columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["name", "company", "fuel_type"])
    ],
    remainder="passthrough"
)

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "car_price_model.pkl")

print("Model trained successfully!")