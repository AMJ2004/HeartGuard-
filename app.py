from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

def load_model():
    try:
        with open("pickle_files/randomf.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Model not found: {e}, using fallback")
        return None

@app.route("/")
def index():
    return render_template("heart.html")

@app.route("/home")
def home():
    return render_template("checkup.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/assessment")
def assessment():
    return render_template("assessment.html")

@app.route("/bmi", methods=["GET", "POST"])
def bmi():
    if request.method == "POST":
        weight = float(request.form.get("weight", 0))
        height = float(request.form.get("height", 0))
        if height > 0:
            bmi_val = round(weight / ((height / 100) ** 2), 2)
            return render_template("bmi.html", bmi=bmi_val)
    return render_template("bmi.html")

@app.route("/diet")
def diet():
    return render_template("diet_preferences.html")

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
def QuitSmoking():
    return render_template("QuitSmoking.html")

@app.route("/quit_smoking")
def quit_smoking():
    return render_template("QuitSmoking.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        sysBP = float(request.form.get("sysBP", 0))
        diaBP = float(request.form.get("diaBP", 0))
        glucose = float(request.form.get("glucose", 0))
        age = int(request.form.get("age", 0))
        bmi_val = float(request.form.get("bmi", 0))
        totChol = float(request.form.get("totChol", 0))
        gender = request.form.get("gender", "0")

        gender_str = "Male" if gender == "1" else "Female"
        p = "Yes" if sysBP > 140 or diaBP > 90 else "No"
        b = "No"
        d = "Yes" if glucose > 126 else "No"

        model = load_model()

        if model:
            data = np.array([[sysBP, glucose, age, totChol, diaBP, 0, 0, 1, 0, bmi_val]])
            prediction = int(model.predict(data)[0])
        else:
            prediction = 0

        if prediction == 1:
            return render_template("heartdisease_detected.html",
                                   age=age, gender=gender_str, bmi=bmi_val,
                                   sysBP=sysBP, diaBP=diaBP, glucose=glucose,
                                   totCHol=totChol, p=p, b=b, d=d)
        else:
            return render_template("nodisease.html",
                                   age=age, gender_str=gender_str, bmi=bmi_val,
                                   sysBP=sysBP, diaBP=diaBP, glucose=glucose,
                                   totChol=totChol, p=p, b=b, d=d)

    except Exception as e:
        return str(e)

@app.route("/diet_recommendations", methods=["POST"])
def diet_recommendations():
    return render_template("diet_results.html",
                           recipes=["Oats", "Green Salad", "Grilled Fish"],
                           health_messages=[
                               "Reduce salt intake",
                               "Exercise daily",
                               "Avoid fried food"
                           ],
                           diet_plan="Heart-friendly low cholesterol diet")

@app.route("/diet_results")
def diet_results():
    return render_template("diet_results.html",
                           recipes=["Salad", "Oats", "Grilled Chicken"],
                           health_messages=["Eat low salt", "Exercise daily"])

if __name__ == "__main__":
    app.run(debug=True)

