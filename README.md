# Auto MPG Fuel Efficiency Prediction Flask App

A Flask web application that predicts vehicle fuel efficiency (MPG) using the Auto MPG dataset.

## Project Objective

This project predicts **Miles Per Gallon (MPG)** using machine learning regression based on vehicle features.

## Features Used

- Cylinders
- Displacement
- Horsepower
- Weight
- Acceleration
- Model Year
- Origin

## Machine Learning Model

- Random Forest Regressor
- R² Score: 0.9116

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML/CSS

## Dataset

Auto MPG Dataset

Target variable:

```text
mpg
```

## How to Run Locally

### Create virtual environment

```bash
python -m venv venv
```

### Activate virtual environment

Windows:

```bash
.\venv\Scripts\Activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Flask app

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Author

Anamika M.