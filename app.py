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

    return jsonify({"prediction": "Diabetic" if prediction == 1 else "Not Diabetic"})

if __name__ == "__main__":
    app.run()