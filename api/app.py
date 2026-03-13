# from flask import Flask, render_template, request
# import pandas as pd
# import joblib
# import os

# import boto3
# from io import BytesIO
# from dotenv import load_dotenv
# import os

# load_dotenv()  

# aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
# aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
# aws_region = os.getenv("AWS_REGION")
# bucket_name = os.getenv("BUCKET_NAME")
# model_object_key = os.getenv("RF_MODEL_OBJECT_KEY")  

# s3 = boto3.client(
#     "s3",
#     aws_access_key_id=aws_access_key,
#     aws_secret_access_key=aws_secret_key,
#     region_name=aws_region
# )

# obj = s3.get_object(Bucket=bucket_name, Key=model_object_key)
# model_buffer = BytesIO(obj['Body'].read())
# data = joblib.load(model_buffer)


# # model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_churn_model.pkl')
# # data = joblib.load(model_path)

# model = data["model"]
# feature_cols = data["features"]
# threshold = data["threshold"]

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route('/predict', methods=['POST'])
# def predict():

#     gender = int(request.form["gender"])
#     senior = int(request.form["SeniorCitizen"])
#     partner = int(request.form["Partner"])
#     tenure = float(request.form["tenure"])
#     monthly = float(request.form["MonthlyCharges"])
#     total = float(request.form["TotalCharges"])

     
#     df = pd.DataFrame(columns=feature_cols)
#     df.loc[0] = 0

    
#     df["gender"] = gender
#     df["SeniorCitizen"] = senior
#     df["Partner"] = partner
#     df["tenure"] = tenure
#     df["MonthlyCharges"] = monthly
#     df["TotalCharges"] = total

#     # Prediction
#     proba = model.predict_proba(df)[0][1]
#     prediction = "High Churn Risk" if proba >= threshold else "Low Churn Risk"

#     return render_template(
#         "index.html",
#         probability=round(proba,3),
#         prediction=prediction,
#         threshold=round(threshold,2)
#     )

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import jsonify
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

import boto3
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()  

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
bucket_name = os.getenv("BUCKET_NAME")
model_object_key = os.getenv("RF_MODEL_OBJECT_KEY")  

s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

obj = s3.get_object(Bucket=bucket_name, Key=model_object_key)
model_buffer = BytesIO(obj['Body'].read())
data = joblib.load(model_buffer)


# model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_churn_model.pkl')
# data = joblib.load(model_path)

model = data["model"]
feature_cols = data["features"]
threshold = data["threshold"]

app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    gender = int(data["gender"])
    senior = int(data["SeniorCitizen"])
    partner = int(data["Partner"])
    tenure = float(data["tenure"])
    monthly = float(data["MonthlyCharges"])
    total = float(data["TotalCharges"])

    df = pd.DataFrame(columns=feature_cols)
    df.loc[0] = 0

    df["gender"] = gender
    df["SeniorCitizen"] = senior
    df["Partner"] = partner
    df["tenure"] = tenure
    df["MonthlyCharges"] = monthly
    df["TotalCharges"] = total

    proba = model.predict_proba(df)[0][1]
    prediction = "High Churn Risk" if proba >= threshold else "Low Churn Risk"

    return jsonify({
        "probability": round(proba,3),
        "prediction": prediction,
        "threshold": round(threshold,2)
    })

if __name__ == "__main__":
    app.run(debug=True)