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
    return render_template("checkup.html")

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

@app.route("/QuitSmoking")
def QuitSmoking():
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
        age = int(request.form.get("age", 0))
        sysBP = float(request.form.get("sysBP", 0))
        diaBP = float(request.form.get("diaBP", 0))
        glucose = float(request.form.get("glucose", 0))
        totChol = float(request.form.get("totChol", 0))
        bmi = float(session.get("bmi", 0))

        # Support both 'gender' (heart.html / assessment.html) and 'male' (home.html)
        gender_val = request.form.get("gender")
        male_val = request.form.get("male")
        if male_val is not None:
            male = int(male_val)
        elif gender_val is not None:
            male = int(gender_val)
        else:
            male = 1

        # Read derived features from form if present (home.html), else derive
        prevalentHyp_val = request.form.get("prevalentHyp")
        if prevalentHyp_val is not None:
            prevalentHyp = int(prevalentHyp_val)
        else:
            prevalentHyp = 1 if sysBP > 140 else 0

        diabetes_val = request.form.get("diabetes")
        if diabetes_val is not None:
            diabetes = int(diabetes_val)
        else:
            diabetes = 1 if glucose > 126 else 0

        bpmeds_val = request.form.get("BPMeds")
        if bpmeds_val is not None:
            BPMeds = int(bpmeds_val)
        else:
            BPMeds = 0

        # Build FULL feature dict
        user_data = {
            "age": age,
            "sysBP": sysBP,
            "diaBP": diaBP,
            "glucose": glucose,
            "totChol": totChol,
            "bmi": bmi,
            "male": male,
            "prevalentHyp": prevalentHyp,
            "diabetes": diabetes,
            "BPMeds": BPMeds,
        }

        # Build feature array in correct order
        # ['sysBP','glucose','age','totChol','diaBP','prevalentHyp','diabetes','male','BPMeds','BMI']
        data = np.array([[
            user_data["sysBP"],
            user_data["glucose"],
            user_data["age"],
            user_data["totChol"],
            user_data["diaBP"],
            user_data["prevalentHyp"],
            user_data["diabetes"],
            user_data["male"],
            user_data["BPMeds"],
            user_data["bmi"]
        ]])

        # Predict with scaler if available
        if model:
            if scaler_available and scaler:
                try:
                    data_scaled = scaler.transform(data)
                    prediction = int(model.predict(data_scaled)[0])
                except Exception as e:
                    print(f"Scaler failed: {e}, falling back to raw features")
                    prediction = int(model.predict(data)[0])
            else:
                prediction = int(model.predict(data)[0])
        else:
            # Fallback heuristic
            prediction = 1 if (sysBP > 140 or totChol > 240 or bmi > 30 or age > 60) else 0

        session["prediction"] = prediction

        # Include prediction in user_data for diet recommender
        user_data["prediction"] = prediction
        session["user_data"] = user_data

        # Helper strings for templates
        gender_str = "Male" if male == 1 else "Female"
        p_str = "Yes" if prevalentHyp == 1 else "No"
        b_str = "Yes" if BPMeds == 1 else "No"
        d_str = "Yes" if diabetes == 1 else "No"

        if prediction == 1:
            return render_template("heartdisease_detected.html",
                                   age=age,
                                   gender=gender_str,
                                   bmi=bmi,
                                   sysBP=sysBP,
                                   diaBP=diaBP,
                                   glucose=glucose,
                                   totCHol=totChol,
                                   p=p_str,
                                   b=b_str,
                                   d=d_str,
                                   feature_rows=[],
                                   high_flags=[],
                                   diet_restrictions=[])
        else:
            return render_template("nodisease.html",
                                   age=age,
                                   gender_str=gender_str,
                                   bmi=bmi,
                                   sysBP=sysBP,
                                   diaBP=diaBP,
                                   glucose=glucose,
                                   totChol=totChol,
                                   p=p_str,
                                   b=b_str,
                                   d=d_str,
                                   feature_rows=[],
                                   chart_uri="",
                                   diet_restrictions=[])

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/diet")
def diet():
    return render_template("diet.html")

@app.route("/diet_results")
def diet_results():
    try:
        user_data = session.get("user_data")

        if not user_data:
            return redirect(url_for("index"))

        df = load_dataset()
        recipes_raw = recommend_recipes(df, user_data)
        recipes = to_recipe_output(recipes_raw)

        diet_plan = personalize_diet(user_data)

        return render_template("diet_results.html",
                               recipes=recipes,
                               diet_plan=diet_plan)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)

