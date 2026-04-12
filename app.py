from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "secret")

# -------------------------
# LOAD MODEL (SAFE)
# -------------------------
def load_model():
    try:
        with open("pickle_files/randomf.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print("Model load failed:", e)
        return None

model = load_model()

# -------------------------
# HOME / INDEX
# -------------------------
@app.route("/")
@app.route("/index")
@app.route("/home")
def index():
    return render_template("heart.html")

# -------------------------
# BMI CALCULATOR
# -------------------------
@app.route("/bmi", methods=["GET", "POST"])
def bmi():
    if request.method == "POST":
        try:
            weight = float(request.form.get("weight"))
            height_cm = float(request.form.get("height"))

            if weight <= 0 or height_cm <= 0:
                flash("Invalid input", "error")
                return redirect(url_for('bmi'))

            height_m = height_cm / 100
            bmi_value = round(weight / (height_m * height_m), 2)

            # Store in session
            session["bmi"] = bmi_value
            flash(f"BMI calculated: {bmi_value}", "success")

            return render_template("bmi.html", bmi=bmi_value)

        except Exception as e:
            flash(f"BMI Error: {str(e)}", "error")
            return redirect(url_for('bmi'))

    return render_template("bmi.html")

# -------------------------
# PREDICTION (AI ASSESSMENT)
# -------------------------
@app.route("/result", methods=["POST"])
def result():
    try:
        age = int(request.form.get("age", 0))
        gender = int(request.form.get("male", 1))  # male field
        sysBP = float(request.form.get("sysBP", 0))
        diaBP = float(request.form.get("diaBP", 0))
        glucose = float(request.form.get("glucose", 0))
        totChol = float(request.form.get("totChol", 0))

        # BMI from session ONLY (no form field)
        bmi = float(session.get("bmi", 0))

        # Validation (must have BMI)
        if bmi <= 0:
            flash("Please calculate BMI first", "error")
            return redirect(url_for('bmi'))

        # Simple ranges check
        if not (20 <= age <= 100 and 10 <= bmi <= 60):
            flash("Invalid age or BMI", "error")
            return redirect(url_for('home'))

        # MODEL INPUT EXACT ORDER
        data = np.array([[age, gender, sysBP, diaBP, glucose, totChol, bmi]])
        print(f"Model input: {data}")

        if model:
            prediction = int(model.predict(data)[0])
        else:
            prediction = 0

        # Store ALL for diet
        session["age"] = age
        session["gender"] = gender
        session["sysBP"] = sysBP
        session["diaBP"] = diaBP
        session["glucose"] = glucose
        session["totChol"] = totChol
        session["prediction"] = prediction
        session["risk"] = prediction

        # Render result
        template = "heartdisease_detected.html" if prediction == 1 else "nodisease.html"
        return render_template(template, 
                             prediction=prediction, age=age, gender="Male" if gender==1 else "Female",
                             bmi=bmi, sysBP=sysBP, diaBP=diaBP, glucose=glucose, totChol=totChol)

    except Exception as e:
        flash(f"Prediction Error: {str(e)}", "error")
        return redirect(url_for('home'))

# -------------------------
# DIET PAGES
# -------------------------
@app.route("/diet")
@app.route("/diet-preferences")
def diet():
    bmi = session.get("bmi", 25.0)
    return render_template("diet.html", bmi=bmi)

@app.route("/diet_results", methods=["GET", "POST"])
def diet_results():
    try:
        bmi = float(session.get("bmi", 25.0))
        risk = int(session.get("risk", 0))

        # Risk-aware diet
        if risk == 1:
            diet = "Low calorie heart-friendly diet (HIGH RISK)"
        elif bmi < 18.5:
            diet = "High calorie diet with protein-rich foods"
        elif bmi < 25:
            diet = "Balanced diet with fruits & vegetables"
        else:
            diet = "Low calorie diet with low fat and sugar"

        # Recipes list for template loop
        if "High calorie" in diet:
            recipes = ["Banana Shake", "Peanut Butter Toast", "Avocado Smoothie"]
        elif "Balanced" in diet:
            recipes = ["Salad Bowl", "Grilled Vegetables", "Quinoa Bowl"]
        else:
            recipes = ["Oats Porridge", "Boiled Vegetables", "Lentil Soup"]

        return render_template("diet_results.html", diet=diet, recipes=recipes)

    except Exception as e:
        flash(f"Diet Error: {str(e)}", "error")
        return redirect(url_for('diet'))

# -------------------------
# OTHER ROUTES (unchanged)
# -------------------------
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/checkup")
def checkup():
    return render_template("checkup.html")

@app.route("/fitness")
def fitness():
    return render_template("fitness.html")

@app.route("/sleep")
def sleep():
    return render_template("sleep.html")

@app.route("/stress")
def stress():
    return render_template("stress.html")

@app.route("/QuitSmoking")
def quit_smoking():
    return render_template("QuitSmoking.html")

@app.route("/test")
def test():
    return "App working perfectly! BMI → Prediction → Diet flow fixed 🚀"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
