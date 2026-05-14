import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------
# LOAD DATASET
# ---------------------------------

df = pd.read_csv("auto-mpg.csv")

# ---------------------------------
# DATA CLEANING
# ---------------------------------

df["horsepower"] = df["horsepower"].replace("?", np.nan)

df["horsepower"] = pd.to_numeric(df["horsepower"])

df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

df.drop_duplicates(inplace=True)

df.drop("car name", axis=1, inplace=True)

# ---------------------------------
# ENCODING
# ---------------------------------

df = pd.get_dummies(df, columns=["origin"], drop_first=True)

# ---------------------------------
# FEATURES AND TARGET
# ---------------------------------

X = df.drop("mpg", axis=1)
y = df["mpg"]

# ---------------------------------
# TRAIN TEST SPLIT
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# FEATURE SCALING
# ---------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------
# MODELS
# ---------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

best_model = None
best_score = -1

print("\nMODEL PERFORMANCE")
print("=" * 50)

for name, model in models.items():

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    predictions = model.predict(X_test_scaled)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Save best model
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# ---------------------------------
# SAVE MODEL
# ---------------------------------

joblib.dump(best_model, "model/model.pkl")

joblib.dump(scaler, "model/scaler.pkl")

print("\n" + "=" * 50)
print(f"Best Model: {best_model_name}")
print(f"Best R2 Score: {best_score:.4f}")
print("Model Saved Successfully!")
