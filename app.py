from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import pickle
import os

app = Flask(__name__)
app.secret_key = "secret"

def load_model():
    try:
        with open("pickle_files/randomf.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

@app.route("/")
def home():
    return render_template("heart.html")

@app.route("/bmi", methods=["GET", "POST"])
def bmi():
    if request.method == "POST":
        weight = float(request.form.get("weight"))
        height_cm = float(request.form.get("height"))

        if weight <= 0 or height_cm <= 0:
            return "Invalid input"

        height_m = height_cm / 100
        bmi_val = round(weight / (height_m ** 2), 2)

        session["bmi"] = bmi_val

        return render_template("bmi.html", bmi=bmi_val)

    return render_template("bmi.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        age = int(request.form.get("age"))
        gender = int(request.form.get("gender"))
        sysBP = float(request.form.get("sysBP"))
        diaBP = float(request.form.get("diaBP"))
        glucose = float(request.form.get("glucose"))
        totChol = float(request.form.get("totChol"))

        bmi = float(session.get("bmi", 0))

        data = np.array([[age, gender, sysBP, diaBP, glucose, totChol, bmi]])

        if model:
            prediction = int(model.predict(data)[0])
        else:
            prediction = 0

        session["prediction"] = prediction

        return render_template("analysis.html", prediction=prediction, bmi=bmi)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/diet")
def diet():
    bmi = float(session.get("bmi", 25))
    prediction = int(session.get("prediction", 0))

    if bmi < 18.5:
        diet = "High calorie diet"
        recipes = ["Banana Shake", "Peanut Butter Toast"]
    elif bmi < 25:
        diet = "Balanced diet"
        recipes = ["Salad", "Grilled Paneer"]
    else:
        diet = "Low calorie heart-friendly diet"
        recipes = ["Oats", "Vegetables"]

    return render_template("diet_results.html", diet=diet, recipes=recipes)

@app.route("/test")
def test():
    return "Working"

if __name__ == "__main__":
    app.run(debug=True)
