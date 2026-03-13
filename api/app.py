from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_churn_model.pkl')
data = joblib.load(model_path)

model = data["model"]
feature_cols = data["features"]
threshold = data["threshold"]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    gender = int(request.form["gender"])
    senior = int(request.form["SeniorCitizen"])
    partner = int(request.form["Partner"])
    tenure = float(request.form["tenure"])
    monthly = float(request.form["MonthlyCharges"])
    total = float(request.form["TotalCharges"])

     
    df = pd.DataFrame(columns=feature_cols)
    df.loc[0] = 0

    
    df["gender"] = gender
    df["SeniorCitizen"] = senior
    df["Partner"] = partner
    df["tenure"] = tenure
    df["MonthlyCharges"] = monthly
    df["TotalCharges"] = total

    # Prediction
    proba = model.predict_proba(df)[0][1]
    prediction = "High Churn Risk" if proba >= threshold else "Low Churn Risk"

    return render_template(
        "index.html",
        probability=round(proba,3),
        prediction=prediction,
        threshold=round(threshold,2)
    )

if __name__ == "__main__":
    app.run(debug=True)