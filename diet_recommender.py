"""Diet recommender module for HeartGuard+"""

import os
import pickle
import numpy as np

# Recipe database
RECIPE_DB = [
    {"name": "Oatmeal with Berries", "calories": 320, "protein": 12, "tags": ["breakfast", "heart_healthy", "low_sodium"]},
    {"name": "Grilled Salmon with Asparagus", "calories": 450, "protein": 38, "tags": ["dinner", "heart_healthy", "omega3"]},
    {"name": "Mediterranean Quinoa Salad", "calories": 380, "protein": 14, "tags": ["lunch", "vegetarian", "heart_healthy"]},
    {"name": "Chicken and Vegetable Stir-Fry", "calories": 420, "protein": 35, "tags": ["dinner", "low_fat"]},
    {"name": "Lentil Soup with Spinach", "calories": 290, "protein": 18, "tags": ["lunch", "vegetarian", "high_fiber"]},
    {"name": "Greek Yogurt with Nuts", "calories": 250, "protein": 15, "tags": ["snack", "protein"]},
    {"name": "Avocado Toast with Egg", "calories": 350, "protein": 16, "tags": ["breakfast", "heart_healthy"]},
    {"name": "Baked Cod with Sweet Potato", "calories": 400, "protein": 32, "tags": ["dinner", "low_sodium"]},
    {"name": "Chickpea and Vegetable Curry", "calories": 360, "protein": 13, "tags": ["dinner", "vegetarian", "high_fiber"]},
    {"name": "Smoothie Bowl with Seeds", "calories": 310, "protein": 10, "tags": ["breakfast", "antioxidant"]},
    {"name": "Turkey and Veggie Lettuce Wraps", "calories": 280, "protein": 26, "tags": ["lunch", "low_carb"]},
    {"name": "Brown Rice and Black Bean Bowl", "calories": 420, "protein": 16, "tags": ["lunch", "vegetarian", "high_fiber"]},
]

# Diet plans by risk/BMI profile
DIET_PLANS = {
    "low_risk_normal": {
        "name": "Balanced Heart-Healthy Diet",
        "description": "Maintain your excellent health with a balanced Mediterranean-style diet.",
        "calories": 2200,
        "focus": ["Whole grains", "Lean proteins", "Healthy fats", "Fruits & vegetables"]
    },
    "low_risk_overweight": {
        "name": "Weight Management Diet",
        "description": "Slightly reduce calories while maintaining heart-healthy nutrition.",
        "calories": 1800,
        "focus": ["Portion control", "High fiber", "Lean proteins", "Reduced sugar"]
    },
    "high_risk_normal": {
        "name": "Strict Heart Care Diet",
        "description": "Intensive heart protection despite normal weight. Focus on sodium reduction and omega-3s.",
        "calories": 2000,
        "focus": ["Very low sodium (<1500mg)", "Omega-3 rich fish", "Plant sterols", "No trans fats"]
    },
    "high_risk_overweight": {
        "name": "Intensive Cardiac Diet",
        "description": "Aggressive weight loss combined with maximum heart protection.",
        "calories": 1500,
        "focus": ["Calorie restriction", "Very low sodium", "Maximum fiber", "Omega-3 supplementation"]
    }
}


def load_dataset():
    """Load or create a minimal dataset for recipe filtering."""
    return RECIPE_DB


def get_minmax_scaler():
    """Try to load a MinMax scaler from pickle_files."""
    try:
        with open("pickle_files/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            return scaler, True
    except Exception as e:
        print(f"Scaler not found: {e}")
        return None, False


def recommend_recipes(dataset, user_data):
    """Recommend recipes based on user health profile."""
    bmi = user_data.get("bmi", 22)
    prediction = user_data.get("prediction", 0)
    sysBP = user_data.get("sysBP", 120)
    glucose = user_data.get("glucose", 100)
    totChol = user_data.get("totChol", 200)

    # Filter recipes based on health markers
    filtered = dataset.copy()

    # High BP -> low sodium recipes
    if sysBP > 130:
        filtered = [r for r in filtered if "low_sodium" in r["tags"]]
    
    # High cholesterol -> heart healthy focus
    if totChol > 200 or prediction == 1:
        filtered = [r for r in filtered if "heart_healthy" in r["tags"]]
        # If no heart healthy recipes, add some
        if len(filtered) < 3:
            filtered = [r for r in dataset if "heart_healthy" in r["tags"]]
    
    # Overweight -> lower calorie
    if bmi > 25:
        filtered = sorted(filtered, key=lambda x: x["calories"])
    else:
        # Normal BMI -> balanced
        filtered = sorted(filtered, key=lambda x: x["calories"], reverse=False)
    
    # Ensure we return at least 6 recipes
    if len(filtered) < 6:
        # Fill with remaining recipes
        remaining = [r for r in dataset if r not in filtered]
        filtered.extend(remaining[:6 - len(filtered)])
    
    return filtered[:6]


def personalize_diet(user_data):
    """Generate personalized diet plan based on user data."""
    bmi = user_data.get("bmi", 22)
    prediction = user_data.get("prediction", 0)

    if prediction == 1 and bmi >= 25:
        return DIET_PLANS["high_risk_overweight"]
    elif prediction == 1:
        return DIET_PLANS["high_risk_normal"]
    elif bmi >= 25:
        return DIET_PLANS["low_risk_overweight"]
    else:
        return DIET_PLANS["low_risk_normal"]


def to_recipe_output(recipes):
    """Convert raw recipe dicts to template-friendly objects with dot notation."""
    class RecipeObj:
        def __init__(self, d):
            self.__dict__.update(d)
    
    return [RecipeObj(r) for r in recipes]
