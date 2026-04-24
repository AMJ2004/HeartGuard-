from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import pickle
import traceback
import os

from diet_recommender import (
    load_dataset,
    get_minmax_scaler,
    recommend_recipes,
    personalize_diet,
    to_recipe_output,
)

app = Flask(__name__)
app.secret_key = "your-super-secret-key-change-in-production"

# ---------------- MODEL ----------------
def load_model():
    try:
        with open("pickle_files/randomf.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Model not found: {e}, using fallback")
        return None

model = load_model()
scaler, scaler_available = get_minmax_scaler()

# ---------------- ERROR HANDLER ----------------
@app.errorhandler(Exception)
def handle_exception(e):
    return f"<h1>Debug: 500 Error</h1><pre>{traceback.format_exc()}</pre><a href='/'>Home</a>", 500

# ---------------- ROUTES ----------------
@app.route("/")
@app.route("/index")
def index():
    return render_template("heart.html")

@app.route("/home")
def home():
    return redirect(url_for('index'))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/stress")
def stress():
    return render_template("stress.html")

@app.route("/fitness")
def fitness():
    return render_template("fitness.html")

@app.route("/sleep")
def sleep():
    return render_template("sleep.html")

@app.route("/quit_smoking")
def quit_smoking():
    return render_template("QuitSmoking.html")

@app.route("/diet_recommendations", methods=["POST"])
def diet_recommendations():
    return redirect(url_for('diet'))

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
        # Collect inputs safely
        age = int(request.form.get("age", 45))
        gender = int(request.form.get("gender", 0))
        sysBP = float(request.form.get("sysBP", 120))
        diaBP = float(request.form.get("diaBP", 80))
        glucose = float(request.form.get("glucose", 100))
        totChol = float(request.form.get("totChol", 200))

        # BMI from session
        bmi = float(session.get("bmi", 0))
        if bmi <= 0:
            return redirect(url_for("bmi"))

        # Derived features for the 10-feature model
        # ['sysBP','glucose','age','totChol','diaBP','prevalentHyp','diabetes','male','BPMeds','BMI']
        prevalentHyp = 1 if sysBP >= 140 else 0
        diabetes = 1 if glucose >= 126 else 0
        male = gender
        BPMeds = 0  # Not collected in form; default to 0

        # Build feature array in correct order
        data = np.array([[sysBP, glucose, age, totChol, diaBP, prevalentHyp, diabetes, male, BPMeds, bmi]])

        # Predict with scaler if available
        if model:
            if scaler_available and scaler:
                try:
                    X_scaled = scaler.transform(data)
                    prediction = int(model.predict(X_scaled)[0])
                except Exception as e:
                    print(f"Scaler failed: {e}, falling back to raw features")
                    prediction = int(model.predict(data)[0])
            else:
                prediction = int(model.predict(data)[0])
        else:
            # Fallback heuristic
            prediction = 1 if (sysBP > 140 or totChol > 240 or bmi > 30 or age > 60) else 0

        # Store in session
        user_data = {
            "age": age,
            "gender": gender,
            "male": male,
            "sysBP": sysBP,
            "diaBP": diaBP,
            "glucose": glucose,
            "totChol": totChol,
            "bmi": bmi,
            "prevalentHyp": prevalentHyp,
            "diabetes": diabetes,
            "BPMeds": BPMeds,
        }
        session["user_data"] = user_data
        session["prediction"] = prediction

        return render_template("analysis.html",
                               prediction=prediction,
                               bmi=bmi)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/diet")
def diet():
    try:
        user_data = session.get("user_data")
        if not user_data:
            # Fallback if session expired
            bmi = float(session.get("bmi", 22))
            prediction = int(session.get("prediction", 0))
            user_data = {"bmi": bmi, "prediction": prediction, "sysBP": 120, "glucose": 100, "totChol": 200}

        # Load dataset and generate recipes
        df = load_dataset()
        recipes_raw = recommend_recipes(df, user_data)
        recipes = to_recipe_output(recipes_raw)

        # Personalize diet plan
        diet_plan = personalize_diet(user_data)

        return render_template("diet_results.html",
                               recipes=recipes,
                               diet_plan=diet_plan,
                               bmi=user_data.get("bmi", 22),
                               risk=user_data.get("prediction", 0))

    except Exception as e:
        print(f"Diet route error: {e}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
