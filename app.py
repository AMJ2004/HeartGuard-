from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import pickle
import traceback
import os

app = Flask(__name__)
app.secret_key = "your-super-secret-key-change-in-production"

def load_model():
    try:
        with open("pickle_files/randomf.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Warning: Model file 'pickle_files/randomf.pkl' not found. Using fallback predictions.")
        return None
    except Exception as e:
        print(f"Model loading error: {e}")
        return None

model = load_model()

@app.errorhandler(Exception)
def handle_exception(e):
    return f"""
    <h1>Debug Info - 500 Error</h1>
    <pre style='background:#f4f4f4;padding:20px;border-radius:8px;overflow:auto;white-space:pre-wrap;'>
{traceback.format_exc()}
    </pre>
    <p><a href="/" class="btn btn-primary">← Home</a></p>
    """, 500

@app.route("/")
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

            if weight <= 0 or height_cm <= 0 or height_cm > 300 or weight > 500:
                return render_template("bmi.html", error="Please enter valid weight (kg) and height (cm)"), 400

            height_m = height_cm / 100
            bmi_val = round(weight / (height_m ** 2), 2)
            session["bmi"] = bmi_val
            session.modified = True
            return render_template("bmi.html", bmi=bmi_val)
        except (ValueError, ZeroDivisionError):
            return render_template("bmi.html", error="Invalid input. Please check your numbers."), 400

    return render_template("bmi.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        # Safe defaults
        age = int(request.form.get("age", 45))
        gender = int(request.form.get("gender", 0)) 
        sysBP = float(request.form.get("sysBP", 120))
        diaBP = float(request.form.get("diaBP", 80))
        glucose = float(request.form.get("glucose", 100))
        totChol = float(request.form.get("totChol", 200))

        bmi = float(session.get("bmi", 22.5))
        if bmi <= 0:
            return redirect(url_for("bmi"))

        data = np.array([[age, gender, sysBP, diaBP, glucose, totChol, bmi]])
        
        if model is not None:
            prediction = int(model.predict(data)[0])
        else:
            prediction = 1 if (sysBP > 140 or totChol > 240 or bmi > 30 or age > 60) else 0
        
        session["prediction"] = prediction
        session.modified = True

        return render_template("analysis.html", prediction=prediction, bmi=bmi)

    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Calculation error: {str(e)}. Please check inputs.", 400

@app.route("/diet")
def diet():
    try:
        bmi = float(session.get("bmi", 22.5))
        prediction = int(session.get("prediction", 0))

        if bmi < 18.5:
            diet_type = "High calorie balanced diet"
            recipes = ["Nutrient-dense Banana Protein Shake", "Peanut Butter & Whole Grain Toast", "Greek Yogurt with Nuts"]
            calories = "2500-3000 kcal/day"
        elif bmi < 25:
            diet_type = "Balanced heart-healthy diet" 
            recipes = ["Quinoa Vegetable Salad", "Grilled Paneer Tikka", "Lentil Soup"]
            calories = "2000-2500 kcal/day"
        elif bmi < 30:
            diet_type = "Weight management diet"
            recipes = ["Oatmeal with Berries", "Steamed Vegetables with Tofu", "Clear Vegetable Broth"]
            calories = "1800-2200 kcal/day"
        else:
            diet_type = "Low calorie heart-friendly diet"
            recipes = ["Oatmeal Porridge", "Mixed Vegetable Stir-fry", "Clear Lentil Soup"]
            calories = "1500-1800 kcal/day"

        if prediction == 1:
            diet_type += " (High risk - cardiac precautions)"
            recipes = [r + " (heart-safe)" for r in recipes]

        return render_template("diet_results.html", 
                             diet=diet_type, 
                             recipes=recipes,
                             calories=calories,
                             bmi=round(bmi, 1),
                             risk=bool(prediction))
    except Exception as e:
        print(f"Diet error: {e}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

