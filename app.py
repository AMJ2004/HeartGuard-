from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import pickle
import traceback
import os

app = Flask(__name__)
app.secret_key = "your-super-secret-key-change-in-production"

# ---------------- MODEL ----------------
def load_model():
    try:
        with open("pickle_files/randomf.pkl", "rb") as f:
            return pickle.load(f)
    except:
        print("Model not found, using fallback")
        return None

model = load_model()

# ---------------- ERROR HANDLER ----------------
@app.errorhandler(Exception)
def handle_exception(e):
    return f"<h1>Debug: 500 Error</h1><pre>{traceback.format_exc()}</pre><a href='/'>Home</a>", 500

# ---------------- ROUTES ----------------
@app.route("/")
@app.route("/home")
@app.route("/index")
def index():
    return render_template("heart.html")

@app.route("/assessment")
def assessment():
    bmi = session.get("bmi", None)
    return render_template("assessment.html", bmi=bmi)

@app.route("/bmi", methods=["GET", "POST"])
def bmi():
    if request.method == "POST":
        try:
            weight = float(request.form.get("weight", 0))
            height_cm = float(request.form.get("height", 0))

            if weight <= 0 or height_cm <= 0:
                return render_template("bmi.html", error="Invalid input")

            height_m = height_cm / 100
            bmi_val = round(weight / (height_m ** 2), 2)

            session["bmi"] = bmi_val
            return render_template("bmi.html", bmi=bmi_val)

        except:
            return render_template("bmi.html", error="Invalid values")

    return render_template("bmi.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        age = int(request.form.get("age", 45))
        gender = int(request.form.get("gender", 0))
        sysBP = float(request.form.get("sysBP", 120))
        diaBP = float(request.form.get("diaBP", 80))
        glucose = float(request.form.get("glucose", 100))
        totChol = float(request.form.get("totChol", 200))

        bmi = float(session.get("bmi", 0))
        if bmi <= 0:
            return redirect(url_for("bmi"))

        data = np.array([[age, gender, sysBP, diaBP, glucose, totChol, bmi]])

        if model:
            prediction = int(model.predict(data)[0])
        else:
            prediction = 1 if (sysBP > 140 or totChol > 240 or bmi > 30 or age > 60) else 0

        session["prediction"] = prediction

        return render_template("analysis.html",
                               prediction=prediction,
                               bmi=bmi)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/diet")
def diet():
    try:
        bmi = float(session.get("bmi", 22))
        prediction = int(session.get("prediction", 0))

        if bmi < 18.5:
            diet = "High calorie diet"
            recipes = ["Banana Shake", "Peanut Butter Toast"]
        elif bmi < 25:
            diet = "Balanced diet"
            recipes = ["Salad", "Grilled Paneer"]
        elif bmi < 30:
            diet = "Weight control diet"
            recipes = ["Oats", "Vegetable Soup"]
        else:
            diet = "Low calorie heart diet"
            recipes = ["Boiled Veggies", "Oats"]

        if prediction == 1:
            diet += " (Heart Risk Care)"

        return render_template("diet_results.html",
                               diet=diet,
                               recipes=recipes,
                               bmi=bmi,
                               risk=prediction)

    except:
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

