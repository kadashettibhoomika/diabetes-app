from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Flask is running successfully"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[
        float(data['pregnancies']),
        float(data['glucose']),
        float(data['bloodpressure']),
        float(data['skinthickness']),
        float(data['insulin']),
        float(data['bmi']),
        float(data['dpf']),
        float(data['age'])
    ]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    confidence = round(max(proba) * 100, 2)

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })