from flask import Flask, render_template, request, session
import os
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "secret")

# Safe imports
try:
    from diet_recommender import *
except:
    pass

try:
    from risk_explainer import *
except:
    pass

try:
    from personalization import *
except:
    pass

try:
    from inference_scaler import *
except:
    pass

try:
    from medical_filters import *
except:
    pass

def load_model():
    try:
        with open("pickle_files/randomf.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

@app.route("/")
def index():
    try:
        return render_template("heart.html")
    except:
        return "HeartGuard App Running 🚀"

@app.route("/test")
def test():
    return "App working perfectly 🚀"

@app.route("/result", methods=["POST"])
def result():
    try:
        sysBP = float(request.form.get("sysBP", 0))
        glucose = float(request.form.get("glucose", 0))
        age = int(request.form.get("age", 0))
        bmi = float(request.form.get("bmi", 0))
        totChol = float(request.form.get("totChol", 0))
        diaBP = float(request.form.get("diaBP", 0))

        model = load_model()

        if model:
            data = np.array([[sysBP, glucose, age, totChol, diaBP, 0, 0, 1, 0, bmi]])
            prediction = model.predict(data)[0]
        else:
            prediction = 0

        return f"Prediction: {prediction}"

    except Exception as e:
        return f"Error: {str(e)}"

@app.errorhandler(Exception)
def handle_error(e):
    return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run()
