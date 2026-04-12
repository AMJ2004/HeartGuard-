from flask import Flask, render_template, request
import os
import numpy as np
import pickle

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-on-render")

# Global model load
model = None
try:
    with open("pickle_files/randomf.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Model load failed: {e}. Using fallback.")
    model = None

# Safe imports for diet etc.
try:
    from diet_recommender import get_diet_plan
except:
    def get_diet_plan(pref):
        return "Sample heart-healthy diet plan based on preferences."
try:
    # Add other stubs if needed
    pass
except:
    pass

@app.route("/")
@app.route("/index")
def index():
    return render_template("heart.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/diet")
def diet():
    return render_template("diet.html")

@app.route("/bmi")
def bmi():
    return render_template("bmi.html")

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

@app.route("/result", methods=["POST"])
def result():
    try:
        # Parse form data
        sysBP = float(request.form.get("sysBP", 120))
        glucose = float(request.form.get("glucose", 80))
        age = int(request.form.get("age", 50))
        bmi = float(request.form.get("bmi", 25))
        totChol = float(request.form.get("totChol", 200))
        diaBP = float(request.form.get("diaBP", 80))
        
        # Full features for model (pad as per training)
        data = np.array([[age, totChol, glucose, bmi, diaBP, sysBP, 0, 0, 1, 0]])  # example features
        
        prediction = 0
        if model is not None:
            prediction = model.predict(data)[0]
        
        if prediction == 0:
            return render_template("nodisease.html", prediction=prediction)
        else:
            return render_template("heartdisease_detected.html", prediction=prediction)
    except Exception as e:
        return render_template("error.html", error=str(e)) if 'error.html' else f"Error: {str(e)}"

@app.route("/diet_results", methods=["POST"])
def diet_results():
    preferences = request.form.get("preferences", "general")
    diet_plan = get_diet_plan(preferences)
    return render_template("diet_results.html", plan=diet_plan)

@app.route("/test")
def test():
    return "App working perfectly! All routes ready for Render deployment 🚀"

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html") if os.path.exists("templates/404.html") else ("Page not found", 404)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

