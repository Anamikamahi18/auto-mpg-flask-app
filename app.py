from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# -----------------------------
# LOAD MODEL AND SCALER
# -----------------------------

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")


# -----------------------------
# HOME PAGE
# -----------------------------


@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# PREDICTION ROUTE
# -----------------------------


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get form data
        cylinders = int(request.form["cylinders"])

        displacement = float(request.form["displacement"])

        horsepower = float(request.form["horsepower"])

        weight = float(request.form["weight"])

        acceleration = float(request.form["acceleration"])

        model_year = int(request.form["model_year"])

        origin = int(request.form["origin"])

        # One-hot encoding
        origin_2 = 0
        origin_3 = 0

        if origin == 2:
            origin_2 = 1

        elif origin == 3:
            origin_3 = 1

        print("\nForm Values:")
        print("Cylinders:", cylinders)
        print("Displacement:", displacement)
        print("Horsepower:", horsepower)
        print("Weight:", weight)
        print("Acceleration:", acceleration)
        print("Model Year:", model_year)
        print("Origin:", origin)
        print("origin_2:", origin_2)
        print("origin_3:", origin_3)

        # Create DataFrame
        features = pd.DataFrame(
            {
                "cylinders": [cylinders],
                "displacement": [displacement],
                "horsepower": [horsepower],
                "weight": [weight],
                "acceleration": [acceleration],
                "model year": [model_year],
                "origin_2": [origin_2],
                "origin_3": [origin_3],
            }
        )
        print("\nFeatures DataFrame:")
        print(features)
        # Scale features
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)[0]

        return render_template(
            "index.html", prediction_text=f"Predicted MPG: {prediction:.2f}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
