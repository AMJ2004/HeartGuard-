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

@app.route("/bmi")
def bmi():
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
        bmi = float(request.form.get("bmi", 0))
        totChol = float(request.form.get("totChol", 0))

        model = load_model()

        if model:
            data = np.array([[sysBP, glucose, age, totChol, diaBP, 0, 0, 1, 0, bmi]])
            prediction = int(model.predict(data)[0])
        else:
            prediction = 0

        if prediction == 1:
            return render_template("heartdisease_detected.html",
                                   age=age, bmi=bmi, sysBP=sysBP,
                                   diaBP=diaBP, glucose=glucose,
                                   totCHol=totChol)
        else:
            return render_template("nodisease.html")

    except Exception as e:
        return str(e)

@app.route("/diet_recommendations", methods=["POST"])
def diet_recommendations():
    try:
        return render_template("diet_results.html",
                               recipes=[],
                               health_messages=["Demo diet plan working"])
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)

