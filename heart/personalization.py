from __future__ import annotations

from typing import Dict, List, Tuple, Any


RiskPriority = [
    "diabetes",
    "high_bp",
    "high_cholesterol",
    "obesity",
]


def _detect_risk_reason(prediction_output: Dict[str, Any]) -> str | None:
    """Infer a single highest-priority risk reason from health indicators.

    prediction_output should contain keys: 'bmi', 'sysBP', 'diaBP', 'glucose',
    optionally 'ldl' or 'totChol', and optionally 'diabetes' flag.
    Thresholds:
      - BMI >= 30 -> obesity
      - sysBP >= 140 or diaBP >= 90 -> high_bp
      - LDL >= 160 (fallback: totChol >= 240) -> high_cholesterol
      - fasting glucose >= 126 or diabetes flag -> diabetes
    Priority: diabetes > high_bp > high_cholesterol > obesity
    """
    bmi = float(prediction_output.get("bmi", 0.0) or 0.0)
    sys_bp = float(prediction_output.get("sysBP", 0.0) or 0.0)
    dia_bp = float(prediction_output.get("diaBP", 0.0) or 0.0)
    glucose = float(prediction_output.get("glucose", 0.0) or 0.0)
    ldl = prediction_output.get("ldl")
    tot_chol = prediction_output.get("totChol")
    diabetes_flag = int(prediction_output.get("diabetes", 0) or 0)

    flags = set()
    if bmi >= 30.0:
        flags.add("obesity")
    if sys_bp >= 140.0 or dia_bp >= 90.0:
        flags.add("high_bp")
    # cholesterol: prefer LDL; fallback to total cholesterol threshold
    if ldl is not None and float(ldl) >= 160.0:
        flags.add("high_cholesterol")
    elif tot_chol is not None and float(tot_chol) >= 240.0:
        flags.add("high_cholesterol")
    if glucose >= 126.0 or diabetes_flag == 1:
        flags.add("diabetes")

    for reason in RiskPriority:
        if reason in flags:
            return reason
    return None


def detect_risk_reason(prediction_output: Dict[str, Any]) -> str | None:
    """Public helper to compute risk reason from health indicators."""
    return _detect_risk_reason(prediction_output)


def _score_recipe_for_constraints(recipe: Dict[str, Any], risk_reason: str | None) -> float:
    """Return a score where higher means better fit.

    Uses available nutrition fields: Calories, SodiumContent, SaturatedFatContent,
    CholesterolContent, CarbohydrateContent, FiberContent, SugarContent, ProteinContent.
    """
    cal = float(recipe.get("Calories", 0) or 0)
    sodium = float(recipe.get("SodiumContent", 0) or 0)
    sat_fat = float(recipe.get("SaturatedFatContent", 0) or 0)
    chol = float(recipe.get("CholesterolContent", 0) or 0)
    carbs = float(recipe.get("CarbohydrateContent", 0) or 0)
    fiber = float(recipe.get("FiberContent", 0) or 0)
    sugar = float(recipe.get("SugarContent", 0) or 0)
    protein = float(recipe.get("ProteinContent", 0) or 0)

    score = 0.0
    # Generic desirables
    score += max(0.0, min(1.0, (fiber - 5) / 10.0))  # prefer higher fiber
    score += max(0.0, min(1.0, (20 - sugar) / 20.0))  # prefer lower sugar

    if risk_reason == "obesity":
        score += max(0.0, min(1.0, (450 - cal) / 450.0))
    elif risk_reason == "high_bp":
        # Daily goal < 1500 mg -> per recipe aim < 500 mg
        score += max(0.0, min(1.0, (500 - sodium) / 500.0))
    elif risk_reason == "high_cholesterol":
        score += max(0.0, min(1.0, (3 - sat_fat) / 3.0))
        score += max(0.0, min(1.0, (75 - chol) / 75.0))
    elif risk_reason == "diabetes":
        score += max(0.0, min(1.0, (35 - carbs) / 35.0))
        score += max(0.0, min(1.0, (10 - sugar) / 10.0))
        score += max(0.0, min(1.0, (fiber - 8) / 12.0))

    return score


def _apply_lifestyle_adjustments(
    recipe: Dict[str, Any],
    *,
    weight_goal: str | None,
    activity_level: str | None,
    alcohol: int | str | None,
    smoking: int | str | None,
    sleep_quality: str | None,
) -> Tuple[float, List[str], float]:
    """Return (score_delta, notes, portion_multiplier)."""
    notes: List[str] = []
    portion_multiplier = 1.0
    score_delta = 0.0

    cal = float(recipe.get("Calories", 0) or 0)
    protein = float(recipe.get("ProteinContent", 0) or 0)
    sugar = float(recipe.get("SugarContent", 0) or 0)

    # Weight goal
    if (weight_goal or '').lower() == 'lose':
        portion_multiplier = 0.8
        score_delta += max(0.0, min(1.0, (450 - cal) / 450.0))
        if sugar > 10:
            score_delta -= 0.3
        notes.append("Weight loss: smaller portions (~20%) and avoid sugary drinks.")
    elif (weight_goal or '').lower() == 'gain':
        portion_multiplier = 1.15
        score_delta += max(0.0, min(1.0, (protein - 15) / 25.0))
        notes.append("Weight gain: include healthy fats (nuts, avocado, olive oil).")

    # Activity level
    alvl = (activity_level or '').lower()
    if alvl in ('sedentary', 'low'):
        score_delta += max(0.0, min(1.0, (450 - cal) / 450.0))
        notes.append("Sedentary: target lower calories; include daily walks.")
    elif alvl in ('active', 'high', 'athlete', 'moderate'):
        score_delta += max(0.0, min(1.0, (protein - 25) / 35.0))
        notes.append("Active: prioritize higher protein for recovery.")

    # Alcohol
    try:
        alc_flag = int(alcohol)
    except Exception:
        alc_flag = 0
    if alc_flag == 1:
        notes.append("Alcohol: reduce to < 2 drinks/week and hydrate well.")

    # Smoking
    try:
        smoke_flag = int(smoking)
    except Exception:
        smoke_flag = 0
    if smoke_flag == 1:
        notes.append("Smoking: add antioxidant-rich foods (berries, leafy greens).")

    # Sleep
    if (sleep_quality or '').lower() in ('poor', 'bad', 'insufficient'):
        notes.append("Sleep: include magnesium-rich foods (bananas, almonds); avoid caffeine at night.")

    return score_delta, notes, portion_multiplier


def _activity_factor(activity_level: str | None) -> float:
    lvl = (activity_level or '').lower()
    if lvl in ('very active', 'very_active', 'high', 'athlete'):  # highest
        return 1.725
    if lvl in ('moderately active', 'moderate', 'medium'):
        return 1.55
    if lvl in ('lightly active', 'light', 'lightly_active'):
        return 1.375
    # sedentary / default
    return 1.2


def _compute_daily_targets(
    *,
    weight_kg: float | None,
    height_cm: float | None,
    age: int | None,
    gender: str | None,
    activity_level: str | None,
    weight_goal: str | None,
) -> tuple[int | None, int | None]:
    """Return (calorie_target, protein_target_g). None if insufficient data.

    BMR (Mifflin–St Jeor):
      male:   10*w + 6.25*h - 5*a + 5
      female: 10*w + 6.25*h - 5*a - 161
    Protein targets (approx midpoints):
      maintain: 1.4 g/kg; lose: 1.9 g/kg; gain: 1.8 g/kg
    """
    try:
        w = float(weight_kg) if weight_kg not in (None, '') else None
        h = float(height_cm) if height_cm not in (None, '') else None
        a = int(age) if age not in (None, '') else None
    except Exception:
        w = h = None
        a = None

    g = (gender or '').lower() if gender is not None else ''
    if w is None or h is None or a is None or not g:
        return None, None

    if g in ('male', 'm', '1', 'man'):
        bmr = 10 * w + 6.25 * h - 5 * a + 5
    else:
        bmr = 10 * w + 6.25 * h - 5 * a - 161

    af = _activity_factor(activity_level)
    tdee = bmr * af

    wg = (weight_goal or '').lower()
    if wg == 'lose':
        tdee -= 500
    elif wg == 'gain':
        tdee += 400

    # Protein target
    if wg == 'lose':
        grams = 1.9 * w
    elif wg == 'gain':
        grams = 1.8 * w
    else:
        grams = 1.4 * w

    # Round outputs
    cal_target = int(round(tdee))
    prot_target = int(round(grams / 5.0) * 5)
    return cal_target, prot_target


def compute_daily_targets(
    *,
    weight_kg: float | None,
    height_cm: float | None,
    age: int | None,
    gender: str | None,
    activity_level: str | None,
    weight_goal: str | None,
) -> tuple[int | None, int | None]:
    """Public wrapper that computes (calorie_target, protein_target_g)."""
    return _compute_daily_targets(
        weight_kg=weight_kg,
        height_cm=height_cm,
        age=age,
        gender=gender,
        activity_level=activity_level,
        weight_goal=weight_goal,
    )


def personalize_diet(
    base_diet: List[Dict[str, Any]],
    prediction_output: Dict[str, Any],
    weight_goal: str | None,
    activity_level: str | None,
    alcohol: int | str | None,
    smoking: int | str | None,
    sleep_quality: str | None,
    *,
    weight_kg: float | None = None,
    height_cm: float | None = None,
    age: int | None = None,
    gender: str | None = None,
    risk_reason_override: str | None = None,
) -> Tuple[List[Dict[str, Any]], str | None, List[str], int | None, int | None]:
    """Merge ML diet with personalized adjustments.

    Returns (final_recipes, risk_reason, notes).
    """
    risk_reason = risk_reason_override or _detect_risk_reason(prediction_output)

    scored: List[Tuple[float, Dict[str, Any], float, List[str]]] = []
    for r in base_diet or []:
        score = _score_recipe_for_constraints(r, risk_reason)
        delta, notes, portion_multiplier = _apply_lifestyle_adjustments(
            r,
            weight_goal=weight_goal,
            activity_level=activity_level,
            alcohol=alcohol,
            smoking=smoking,
            sleep_quality=sleep_quality,
        )
        total = score + delta
        r = dict(r)  # copy
        r['personalization_notes'] = notes
        r['portion_multiplier'] = portion_multiplier
        r['fit_score'] = round(total, 3)
        scored.append((total, r, portion_multiplier, notes))

    # sort by score descending and keep top 20
    scored.sort(key=lambda t: t[0], reverse=True)
    final_recipes = [r for _, r, _, _ in scored][:20]

    # Add risk-reason-specific global note
    global_notes: List[str] = []
    if risk_reason == 'obesity':
        global_notes.append('Focus on high-fiber, lower-calorie meals to support a ~500 kcal/day deficit.')
    elif risk_reason == 'high_bp':
        global_notes.append('Follow DASH: keep sodium low and emphasize fruits/vegetables and whole grains.')
    elif risk_reason == 'high_cholesterol':
        global_notes.append('Favor whole grains, plant proteins, and omega-3 sources; limit saturated fat.')
    elif risk_reason == 'diabetes':
        global_notes.append('Prioritize low-GI, high-fiber meals; avoid refined sugars.')

    # Compute daily targets (calories & protein)
    cal_target, prot_target = _compute_daily_targets(
        weight_kg=weight_kg,
        height_cm=height_cm,
        age=age,
        gender=gender,
        activity_level=activity_level,
        weight_goal=weight_goal,
    )

    return final_recipes, risk_reason, global_notes, cal_target, prot_target


