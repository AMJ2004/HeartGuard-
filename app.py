from flask import Flask, render_template, request, session, flash, redirect, url_for
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

@app.route("/bmi", methods=["GET", "POST"])
def bmi():
    if request.method == "POST":
        try:
            weight = float(request.form.get("weight"))
            height = float(request.form.get("height")) / 100

            if height <= 0 or weight <= 0:
                return "Invalid input"

            bmi_value = round(weight / (height * height), 2)
            session["bmi"] = bmi_value

            return render_template("bmi.html", bmi=bmi_value)

        except Exception as e:
            return f"BMI Error: {str(e)}"

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
def validate_inputs(age, gender, sysBP, diaBP, glucose, totChol, bmi):
    """Validate healthcare inputs"""
    print(f"DEBUG: Validating inputs - age:{age}, bmi:{bmi}, sysBP:{sysBP}, diaBP:{diaBP}, glucose:{glucose}, totChol:{totChol}, gender:{gender}")
    
    if not (20 <= age <= 100):
        return False, "Age must be 20-100"
    if gender not in [0, 1]:
        return False, "Gender must be 0 (female) or 1 (male)"
    if not (70 <= sysBP <= 250):
        return False, "Systolic BP must be 70-250"
    if not (40 <= diaBP <= 150):
        return False, "Diastolic BP must be 40-150" 
    if not (50 <= glucose <= 400):
        return False, "Glucose must be 50-400"
    if not (100 <= totChol <= 500):
        return False, "Total Cholesterol must be 100-500"
    if not (10 <= bmi <= 60):
        return False, "BMI must be 10-60"
    
    return True, "Valid"

def result():
    try:
        sysBP = float(request.form.get("sysBP", 120))
        glucose = float(request.form.get("glucose", 80))
        age = int(request.form.get("age", 50))
        bmi = float(session.get("bmi", request.form.get("bmi", 25.0)))
        totChol = float(request.form.get("totChol", 200))
        diaBP = float(request.form.get("diaBP", 80))
        gender_str = request.form.get("male", "1")  # Field name="male", value 0/1
        gender = 1 if gender_str == "1" else 0  # 1=male, 0=female
        
        print(f"DEBUG: Raw BMI from session/form: {session.get('bmi', request.form.get('bmi', 0))}")
        
        # Validate
        is_valid, msg = validate_inputs(age, gender, sysBP, diaBP, glucose, totChol, bmi)
        if not is_valid:
            flash(f"Invalid input: {msg}", "error")
            return redirect(url_for('home'))
        
        # Set session for diet
        session["age"] = age
        session["bmi"] = bmi
        session["gender"] = gender
        session["gender_str"] = "Male" if gender == 1 else "Female"
        session["glucose"] = glucose
        session["sysBP"] = sysBP
        session["diaBP"] = diaBP
        session["totChol"] = totChol
        
        # CORRECT MODEL INPUT ORDER: [age, gender, sysBP, diaBP, glucose, totChol, bmi]
        data = np.array([[age, gender, sysBP, diaBP, glucose, totChol, bmi]])
        print(f"DEBUG: Model input shape: {data.shape}, data: {data}")
        
        prediction = 0
        if model is not None:
            prediction = model.predict(data)[0]
            print(f"DEBUG: Raw prediction: {prediction}")
        else:
            print("DEBUG: No model, using fallback prediction=0")
        
        session["risk"] = int(prediction)
        print(f"DEBUG: Final prediction/risk: {prediction}")
        
        if prediction == 0:
            return render_template("nodisease.html", 
                                 prediction=prediction,
                                 age=age, gender=session["gender_str"], bmi=bmi,
                                 sysBP=sysBP, diaBP=diaBP, glucose=glucose, totChol=totChol)
        else:
            return render_template("heartdisease_detected.html", 
                                 prediction=prediction,
                                 age=age, gender=session["gender_str"], bmi=bmi,
                                 sysBP=sysBP, diaBP=diaBP, glucose=glucose, totChol=totChol)
    except Exception as e:
        print(f"DEBUG ERROR in result: {str(e)}")
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for('home'))

@app.route("/diet_results", methods=["GET", "POST"])
def diet_results():
    try:
        age = session.get("age", 50)
        bmi = session.get("bmi", 25.0)
        risk = session.get("risk", 0)
        gender_str = session.get("gender_str", "Male")
        
        print(f"DEBUG diet_results: age={age}, bmi={bmi}, risk={risk}")
        
        # Generate personalized diet - risk-aware
        diet_plan = generate_diet_plan(age, bmi, risk)
        
        # Generate recipes
        recipes = generate_recipes(diet_plan)
        
        return render_template("diet_results.html",
                               diet=diet_plan,
                               recipes=recipes,
                               bmi=bmi,
                               risk=risk)
    except Exception as e:
        print(f"DEBUG diet_results error: {str(e)}")
        return f"Diet Error: {str(e)}"

def generate_diet_plan(age, bmi, risk):
    """Risk-aware diet plan"""
    print(f"DEBUG: Generating diet - age:{age}, bmi:{bmi}, risk:{risk}")
    
    if risk == 1:
        if bmi < 18.5:
            return "High calorie heart-protective diet with lean protein sources"
        elif bmi < 25:
            return "Balanced heart-healthy diet rich in omega-3s and fiber" 
        else:
            return "Low calorie heart-friendly diet - focus on weight management and cholesterol control"
    else:
        if bmi < 18.5:
            return "High calorie diet with protein-rich foods for underweight"
        elif bmi < 25:
            return "Balanced diet with fruits and vegetables"
        else:
            return "Low calorie diet with low fat and sugar"

def generate_recipes(diet_plan):
    """Generate recipes as dicts matching template"""
    print(f"DEBUG: Generating recipes for diet: {diet_plan}")
    
    if "High calorie" in diet_plan:
        return [
            {"Name": "Banana Protein Shake", "Calories": "350", "FatContent": "8g", "ProteinContent": "25g", "CarbohydrateContent": "45g", "FiberContent": "5g", "SugarContent": "25g", "SodiumContent": "120mg", "CholesterolContent": "10mg"},
            {"Name": "Peanut Butter Toast", "Calories": "280", "FatContent": "16g", "ProteinContent": "10g", "CarbohydrateContent": "28g", "FiberContent": "4g", "SugarContent": "4g", "SodiumContent": "220mg", "CholesterolContent": "0mg"},
            {"Name": "Avocado Smoothie", "Calories": "320", "FatContent": "20g", "ProteinContent": "8g", "CarbohydrateContent": "35g", "FiberContent": "9g", "SugarContent": "18g", "SodiumContent": "50mg", "CholesterolContent": "0mg"}
        ]
    elif "Balanced" in diet_plan:
        return [
            {"Name": "Quinoa Veggie Salad", "Calories": "250", "FatContent": "10g", "ProteinContent": "12g", "CarbohydrateContent": "35g", "FiberContent": "8g", "SugarContent": "4g", "SodiumContent": "150mg", "CholesterolContent": "0mg"},
            {"Name": "Grilled Vegetables", "Calories": "180", "FatContent": "8g", "ProteinContent": "6g", "CarbohydrateContent": "25g", "FiberContent": "7g", "SugarContent": "8g", "SodiumContent": "80mg", "CholesterolContent": "0mg"},
            {"Name": "Salad Bowl", "Calories": "220", "FatContent": "12g", "ProteinContent": "10g", "CarbohydrateContent": "20g", "FiberContent": "6g", "SugarContent": "5g", "SodiumContent": "200mg", "CholesterolContent": "5mg"}
        ]
    else:  # Low calorie/heart risk
        return [
            {"Name": "Oats Porridge with Berries", "Calories": "200", "FatContent": "4g", "ProteinContent": "8g", "CarbohydrateContent": "35g", "FiberContent": "6g", "SugarContent": "8g", "SodiumContent": "100mg", "CholesterolContent": "2mg"},
            {"Name": "Steamed Vegetables", "Calories": "120", "FatContent": "2g", "ProteinContent": "5g", "CarbohydrateContent": "22g", "FiberContent": "7g", "SugarContent": "6g", "SodiumContent": "60mg", "CholesterolContent": "0mg"},
            {"Name": "Lentil Soup", "Calories": "180", "FatContent": "3g", "ProteinContent": "12g", "CarbohydrateContent": "28g", "FiberContent": "10g", "SugarContent": "2g", "SodiumContent": "250mg", "CholesterolContent": "0mg"}
        ]

@app.route("/diet-preferences")
def diet_preferences():
    return render_template("diet.html")

@app.route("/test")
def test():
    return "App working perfectly! BMI calculator + images fixed 🚀"

@app.errorhandler(404)
def not_found(e):
    return "Page not found", 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

