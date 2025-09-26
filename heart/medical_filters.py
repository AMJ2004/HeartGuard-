from typing import List, Dict


# Catalog of restrictions with reasons per condition.
# This is designed for both UI explanations and programmatic filtering.
# Keys are normalized condition names: 'diabetes', 'high_bp' (hypertension), 'cholesterol'.
RESTRICTION_CATALOG: Dict[str, List[Dict[str, str]]] = {
    'diabetes': [
        {'item': 'white bread', 'why': 'Refined carbs spike blood sugar'},
        {'item': 'white rice', 'why': 'High glycemic index increases glucose quickly'},
        {'item': 'pastries', 'why': 'High in sugar and refined flour'},
        {'item': 'cakes', 'why': 'High sugar content and refined flour'},
        {'item': 'candies', 'why': 'Concentrated sugar spikes glucose'},
        {'item': 'sugary drinks', 'why': 'Liquid sugar rapidly elevates blood sugar'},
        {'item': 'ice cream', 'why': 'High in sugar; can spike glucose'},
        {'item': 'sweetened breakfast cereals', 'why': 'Added sugars raise insulin needs'},
        {'item': 'flavored yogurt', 'why': 'Often has added sugar; choose unsweetened'},
        {'item': 'watermelon', 'why': 'High-glycemic fruit; limit portion size'},
        {'item': 'pineapple', 'why': 'High-glycemic fruit; limit portion size'},
        {'item': 'ripe bananas', 'why': 'Higher sugar content when very ripe; small portions only'},
    ],
    'high_bp': [
        {'item': 'table salt', 'why': 'Excess sodium raises blood pressure'},
        {'item': 'processed meats (bacon, ham, salami)', 'why': 'Very high in sodium and preservatives'},
        {'item': 'canned soups', 'why': 'Often high in sodium'},
        {'item': 'packaged snacks', 'why': 'Typically high sodium content'},
        {'item': 'instant noodles', 'why': 'Seasoning packets contain high sodium'},
        {'item': 'pickles', 'why': 'High in sodium due to brine'},
        {'item': 'soy sauce', 'why': 'Extremely high sodium; use low-sodium alternatives'},
        {'item': 'pizza', 'why': 'High sodium and saturated fat'},
        {'item': 'burgers', 'why': 'Often high in sodium and saturated fat'},
        {'item': 'fries', 'why': 'Fried and salted; increases sodium intake'},
        {'item': 'fried chicken', 'why': 'High sodium and unhealthy fats'},
        {'item': 'energy drinks', 'why': 'Caffeine can temporarily spike blood pressure'},
        {'item': 'strong coffee', 'why': 'Excess caffeine can elevate blood pressure'},
        {'item': 'alcohol (excess)', 'why': 'Raises blood pressure and affects medication'},
    ],
    'cholesterol': [
        {'item': 'butter', 'why': 'High in saturated fat; raises LDL cholesterol'},
        {'item': 'ghee', 'why': 'Concentrated milk fat; high in saturated fat'},
        {'item': 'cream', 'why': 'High saturated fat increases LDL'},
        {'item': 'fatty cuts of red meat', 'why': 'Saturated fat raises LDL and heart risk'},
        {'item': 'full-fat cheese', 'why': 'High in saturated fat; raises LDL'},
        {'item': 'margarine (partially hydrogenated)', 'why': 'Trans fats increase LDL and lower HDL'},
        {'item': 'commercial baked goods', 'why': 'Often contain trans fats and added sugars'},
        {'item': 'fried snacks', 'why': 'Trans fats and oxidized oils harm cardiovascular health'},
    ],
}


def filter_restricted_foods(user_conditions: List[str], recipe_list: List[Dict]) -> List[Dict]:
    """Return recipes that do NOT contain restricted items for the given conditions.

    Parameters
    ----------
    user_conditions: list[str]
        Examples: ["diabetes"], ["high_bp"], ["diabetes", "high_bp"], ["cholesterol"].
        Matching is case-insensitive; synonyms like "hypertension" are mapped to "high_bp".
    recipe_list: list[dict]
        Each recipe is a dict with keys:
          - "name": str
          - "ingredients": list[str]

    Behavior
    --------
    - Build a union of restricted keywords across selected conditions
    - Case-insensitive substring match against each ingredient
    - If any restricted keyword is present, the recipe is excluded
    - Returns a new list preserving the original order
    """

    # 1) Normalize conditions and handle synonyms
    normalized: List[str] = []
    for c in user_conditions or []:
        c_norm = str(c or '').strip().lower()
        if c_norm in ('hypertension', 'high blood pressure', 'bp', 'highbp'):
            c_norm = 'high_bp'
        normalized.append(c_norm)

    # 2) Build the set of restricted terms for selected conditions
    restricted_terms: set[str] = set()
    for c in normalized:
        if c in RESTRICTION_CATALOG:
            for entry in RESTRICTION_CATALOG[c]:
                term = entry.get('item', '').strip().lower()
                if term:
                    restricted_terms.add(term)

    # 3) If no conditions, return as-is
    if not restricted_terms:
        return list(recipe_list or [])

    # 4) Helper: check if a recipe contains any restricted term
    def contains_restricted(ingredients: List[str]) -> bool:
        for raw in ingredients or []:
            try:
                ing = str(raw).lower()
            except Exception:
                continue
            for term in restricted_terms:
                if term in ing:
                    return True
        return False

    # 5) Filter recipes
    allowed: List[Dict] = []
    for recipe in recipe_list or []:
        ings = recipe.get('ingredients') or []
        if not contains_restricted(ings):
            allowed.append(recipe)
    return allowed


__all__ = [
    'RESTRICTION_CATALOG',
    'filter_restricted_foods',
]


