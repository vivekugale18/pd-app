from flask import Flask, render_template, request
import numpy as np
import librosa
import joblib
import torch

from preprocessing import extract_features, segment_audio
from federated_model import ParkinsonNN

app = Flask(__name__)

# Load models
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

fl_model = ParkinsonNN()
fl_model.load_state_dict(torch.load("federated_model.pth"))
fl_model.eval()


# --------------------------
# ROUTES
# --------------------------

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/detect', methods=["GET", "POST"])
def detect():

    if request.method == "POST":

        file = request.files['file']
        model_choice = request.form.get("model")

        filepath = "temp.wav"
        file.save(filepath)

        segments = segment_audio(filepath)

        predictions = []
        probabilities = []

        for segment in segments:
            features = extract_features(segment)
            features = scaler.transform([features])

            if model_choice == "svm":
                pred = svm_model.predict(features)[0]
                prob = svm_model.predict_proba(features)[0]

            else:
                features_tensor = torch.tensor(features).float()
                with torch.no_grad():
                    output = fl_model(features_tensor)
                    prob = torch.softmax(output, dim=1).numpy()[0]
                    pred = np.argmax(prob)

            predictions.append(pred)
            probabilities.append(prob)

        avg_prob = np.mean(probabilities, axis=0)
        final_pred = np.argmax(avg_prob)
        confidence = np.max(avg_prob)

        result = "Parkinson’s Detected" if final_pred == 1 else "Healthy"

        return render_template("result.html",
                               result=result,
                               confidence=round(confidence * 100, 2))

    return render_template("detect.html")


@app.route('/federated')
def federated():
    return render_template("federated.html")


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)