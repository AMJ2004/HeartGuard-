import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


_DATAFRAME_CACHE: Optional[pd.DataFrame] = None


def _read_csv_autodetect(path: str) -> pd.DataFrame:
    """Read CSV handling common compression cases.

    - First, try compression='gzip'
    - If that fails, try compression='infer'
    - As a final fallback, detect gzip by magic bytes and retry with compression='gzip'
    """
    try:
        return pd.read_csv(path, compression='gzip')
    except Exception:
        try:
            return pd.read_csv(path, compression='infer')
        except Exception:
            # Detect gzip by magic number 0x1f 0x8b
            try:
                with open(path, 'rb') as fh:
                    head = fh.read(2)
                if head == b'\x1f\x8b':
                    return pd.read_csv(path, compression='gzip')
            except Exception:
                pass
            # Blind retry as gzip
            return pd.read_csv(path, compression='gzip')


def load_dataset(preferred_paths: Optional[List[str]] = None) -> pd.DataFrame:
    """Load the recipes dataset (gzipped CSV) and cache it.

    Tries provided absolute paths first, then common relative fallbacks.
    """
    global _DATAFRAME_CACHE
    if _DATAFRAME_CACHE is not None:
        return _DATAFRAME_CACHE

    candidate_paths = preferred_paths or []
    candidate_paths += [
        os.path.join('Data', 'dataset.csv'),
        os.path.join('..', 'Diet-Recommendation-System-main', 'Data', 'dataset.csv'),
    ]
    # Prioritize local Data/dataset.csv
    if os.path.join('Data', 'dataset.csv') not in candidate_paths:
        candidate_paths.insert(0, os.path.join('Data', 'dataset.csv'))

    last_error: Optional[Exception] = None
    for path in candidate_paths:
        try:
            df = _read_csv_autodetect(path)
            _DATAFRAME_CACHE = df
            return df
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Unable to load dataset.csv from any path. Last error: {last_error}")


def _scaling(dataframe: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    prepared = scaler.fit_transform(dataframe.iloc[:, 6:15].to_numpy())
    return prepared, scaler


def _nn_predictor(prepared: np.ndarray) -> NearestNeighbors:
    neigh = NearestNeighbors(metric='cosine', algorithm='brute')
    neigh.fit(prepared)
    return neigh


def _build_pipeline(neigh: NearestNeighbors, scaler: StandardScaler, params: dict) -> Pipeline:
    transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
    pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])
    return pipeline


def _extract_includes(dataframe: pd.DataFrame, include_ingredients: List[str]) -> pd.DataFrame:
    if not include_ingredients:
        return dataframe
    extracted = dataframe.copy()
    regex_string = ''.join(map(lambda x: f'(?=.*{re.escape(x)})', include_ingredients))
    return extracted[extracted['RecipeIngredientParts'].str.contains(regex_string, regex=True, flags=re.IGNORECASE)]


def _apply_excludes(dataframe: pd.DataFrame, exclude_ingredients: List[str]) -> pd.DataFrame:
    if not exclude_ingredients:
        return dataframe
    filtered = dataframe.copy()
    pattern = '|'.join(re.escape(x) for x in exclude_ingredients)
    mask = ~filtered['RecipeIngredientParts'].str.contains(pattern, regex=True, flags=re.IGNORECASE)
    return filtered[mask]


def recommend_recipes(
    dataset: pd.DataFrame,
    nutrition_input: List[float],
    include_ingredients: Optional[List[str]] = None,
    exclude_ingredients: Optional[List[str]] = None,
    params: Optional[dict] = None,
):
    include_ingredients = include_ingredients or []
    exclude_ingredients = exclude_ingredients or []
    params = params or {'n_neighbors': 5, 'return_distance': False}

    extracted = _extract_includes(dataset, include_ingredients)
    extracted = _apply_excludes(extracted, exclude_ingredients)

    if extracted.shape[0] >= params['n_neighbors']:
        prepared, scaler = _scaling(extracted)
        neigh = _nn_predictor(prepared)
        pipeline = _build_pipeline(neigh, scaler, params)
        _input = np.array(nutrition_input).reshape(1, -1)
        indices = pipeline.transform(_input)[0]
        return extracted.iloc[indices]
    else:
        return None


def _extract_quoted_strings(s: str) -> List[str]:
    return re.findall(r'"([^"]*)"', s or '')


def to_recipe_output(dataframe: Optional[pd.DataFrame]):
    if dataframe is None:
        return None
    output = dataframe.copy().to_dict('records')
    for recipe in output:
        # Standardize fields if present
        if 'RecipeIngredientParts' in recipe:
            recipe['RecipeIngredientParts'] = _extract_quoted_strings(recipe['RecipeIngredientParts'])
        if 'RecipeInstructions' in recipe:
            recipe['RecipeInstructions'] = _extract_quoted_strings(recipe['RecipeInstructions'])
    return output


def filter_restricted_foods(user_conditions: List[str], recipe_list: List[dict]) -> List[dict]:
    """Filter out recipes containing restricted ingredients for the given conditions.

    This implements the combined restriction table for Diabetes and High Blood Pressure.

    Args:
        user_conditions: list like ["diabetes"], ["high_bp"], or both.
        recipe_list: list of recipe dicts with either "RecipeIngredientParts" (our dataset)
                     or a generic "ingredients" list.

    Returns:
        A new list with recipes that are safe/allowed, preserving the input order.

    Matching is case-insensitive and uses substring contains.
    """

    # 1) Normalize condition names
    conditions = {str(c).strip().lower() for c in (user_conditions or [])}

    # 2) Build restriction keywords per the combined table
    #    We store representative ingredient keywords that commonly appear in recipe datasets.
    diabetes_terms = [
        # Refined carbs & sugary foods
        'white bread', 'white rice', 'pastry', 'cake', 'candy', 'sugary drink', 'soft drink', 'soda', 'ice cream',
        # Sweetened cereals & yogurts
        'sweetened cereal', 'sweetened oatmeal', 'flavored yogurt', 'sweetened yogurt',
        # High-glycemic fruits in excess (flag explicitly)
        'watermelon', 'pineapple', 'ripe banana',
        # Generic sugar terms
        'added sugar', 'granulated sugar', 'syrup', 'corn syrup', 'brown sugar', 'confectioners sugar',
    ]
    high_bp_terms = [
        # High sodium
        'table salt', 'salt', 'soy sauce', 'pickle', 'pickles', 'brine', 'bouillon', 'stock cube', 'monosodium glutamate', 'msg',
        # Processed meats
        'bacon', 'ham', 'salami', 'sausage', 'pepperoni', 'deli meat', 'prosciutto',
        # Processed & fast foods
        'pizza', 'burger', 'fries', 'fried chicken', 'instant noodle', 'instant noodles', 'ramen',
        # Caffeine excess (flag high-caffeine beverages)
        'energy drink', 'espresso', 'strong coffee',
        # Alcohol (excess)
        'beer', 'whiskey', 'vodka', 'rum', 'cocktail',
    ]
    shared_terms = [
        # High-saturated fat foods
        'butter', 'ghee', 'heavy cream', 'whipping cream', 'full-fat cheese', 'fatty beef', 'fatty pork', 'fatty lamb',
        # Trans-fat containing foods (approximate keywords)
        'margarine', 'shortening', 'hydrogenated',
        # Processed/fried snacks & meals
        'fried', 'deep-fried', 'fast food', 'frozen meal',
        # Sweetened breakfast cereals (general catch)
        'sweetened cereal',
    ]

    # 3) Map our accepted condition keys
    condition_to_terms = {
        'diabetes': diabetes_terms + shared_terms,
        'high_bp': high_bp_terms + shared_terms,
        # Backwards-compat names used elsewhere in code (just in case)
        'hypertension': high_bp_terms + shared_terms,
        'cholesterol': [
            # Emphasize high sat/trans fat and fatty cuts
            'red meat', 'lard', 'butter', 'ghee', 'heavy cream', 'full-fat cheese',
            'fatty beef', 'fatty pork', 'fatty lamb', 'margarine', 'shortening', 'hydrogenated', 'fried'
        ] + shared_terms,
    }

    # 4) Union the restricted terms for the selected conditions
    restricted_terms: set[str] = set()
    for key, terms in condition_to_terms.items():
        if key in conditions:
            for t in terms:
                if t and isinstance(t, str):
                    restricted_terms.add(t.lower())

    # 5) If no conditions, return original list
    if not restricted_terms:
        return list(recipe_list or [])

    # 6) Helper: check if recipe contains any restricted term (case-insensitive, substring)
    def recipe_has_restricted_ingredient(recipe: dict) -> bool:
        # Prefer dataset-specific field; fallback to generic
        ingredients = (
            recipe.get('RecipeIngredientParts')
            or recipe.get('ingredients')
            or []
        )
        # Normalize once
        norm_ings: List[str] = []
        for ing in ingredients:
            try:
                norm_ings.append(str(ing).lower())
            except Exception:
                continue
        # Substring match any restricted term
        for ing in norm_ings:
            for term in restricted_terms:
                if term in ing:
                    return True
        return False

    # 7) Filter: keep only recipes without restricted ingredients
    allowed: List[dict] = []
    for r in recipe_list or []:
        if not recipe_has_restricted_ingredient(r):
            allowed.append(r)
    return allowed


def build_nutrition_target(
    *,
    bmi: float,
    tot_chol: float,
    diabetes: int,
    hypertension: int,
    alcohol: int,
    protein_target_g: float | None = None,
    calorie_target_kcal: float | None = None,
) -> List[float]:
    """Return nutrition target vector: [Calories, Fat, SatFat, Cholesterol, Sodium, Carbs, Fiber, Sugar, Protein].

    If daily targets are provided, we map to per-meal values by dividing by 3.
    """
    calories = 400.0
    fat = 12.0
    sat_fat = 3.0
    cholesterol = 30.0
    sodium = 400.0
    carbs = 45.0
    fiber = 8.0
    sugar = 8.0
    protein = 20.0

    if bmi >= 24.5:
        calories = 350.0
        fat = max(8.0, fat - 2.0)

    if tot_chol >= 240.0:
        sat_fat = 2.0
        cholesterol = 20.0
        fat = min(fat, 10.0)

    if diabetes == 1:
        carbs = 30.0
        sugar = 5.0
        fiber = 12.0

    if hypertension == 1:
        sodium = 300.0

    if alcohol == 1:
        # No direct numeric mapping; bias towards higher protein to support satiety
        protein = max(protein, 22.0)

    # Override with per-meal targets when provided
    if calorie_target_kcal is not None and calorie_target_kcal > 0:
        calories = max(200.0, float(calorie_target_kcal) / 3.0)
    if protein_target_g is not None and protein_target_g > 0:
        protein = max(15.0, float(protein_target_g) / 3.0)

    return [calories, fat, sat_fat, cholesterol, sodium, carbs, fiber, sugar, protein]


def build_exclude_list_for_diet_type(diet_type: str) -> List[str]:
    diet_type = (diet_type or '').lower()
    if diet_type == 'vegan':
        return ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'egg', 'milk', 'cheese', 'butter', 'yogurt', 'honey']
    if diet_type in ('vegetarian', 'veg'):
        return ['chicken', 'beef', 'pork', 'fish', 'shrimp']
    # non-veg → no exclusions
    return []


