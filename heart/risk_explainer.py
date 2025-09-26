from __future__ import annotations

from typing import Dict, List, Tuple
import io
import base64

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_risk_contributions(features: Dict[str, float]) -> List[Tuple[str, float]]:
    """Compute heuristic risk contributions (0..1) for each feature.

    Expected keys: sysBP, diaBP, glucose, age, totChol, prevalentHyp, diabetes, BPMeds, bmi
    """
    sysBP = float(features.get('sysBP', 0))
    diaBP = float(features.get('diaBP', 0))
    glucose = float(features.get('glucose', 0))
    age = float(features.get('age', 0))
    totChol = float(features.get('totChol', 0))
    prevalentHyp = int(features.get('prevalentHyp', 0))
    diabetes = int(features.get('diabetes', 0))
    bpmeds = int(features.get('BPMeds', 0))
    bmi = float(features.get('bmi', 0))

    def clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    contributions: List[Tuple[str, float]] = []

    contributions.append((
        'Systolic BP',
        clip01((sysBP - 120.0) / 80.0)
    ))
    contributions.append((
        'Diastolic BP',
        clip01((diaBP - 80.0) / 40.0)
    ))
    contributions.append((
        'Fasting Glucose',
        clip01((glucose - 100.0) / 150.0)
    ))
    contributions.append((
        'Age',
        clip01((age - 45.0) / 35.0)
    ))
    contributions.append((
        'Total Cholesterol',
        clip01((totChol - 200.0) / 100.0)
    ))
    contributions.append((
        'BMI',
        clip01((bmi - 24.5) / 10.0)
    ))
    contributions.append((
        'Hypertension (history)',
        0.7 if prevalentHyp == 1 else 0.0
    ))
    contributions.append((
        'Diabetes (diagnosed)',
        0.9 if diabetes == 1 else 0.0
    ))
    contributions.append((
        'BP Medication',
        0.3 if bpmeds == 1 else 0.0
    ))

    # Sort descending by contribution
    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions


def contributions_to_data_uri(contributions: List[Tuple[str, float]], *, title: str = 'Top contributing factors') -> str:
    # Keep top 6 for readability
    top = contributions[:6]
    labels = [name for name, _ in top][::-1]
    values = [score for _, score in top][::-1]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.barh(labels, values, color='#c0392b')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Relative contribution (0..1)')
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{b64}"


def get_recommended_targets(*, age: int | None, weight_kg: float | None, bmi: float | None,
                            has_diabetes: int | None, has_hypertension: int | None,
                            total_chol: float | None) -> dict:
    """Return recommended upper/lower bounds for key metrics.

    Uses widely accepted adult targets; simplifies by age bands.
    """
    # Blood pressure targets
    if age is not None and age >= 60:
        sysbp_target = 130.0
    else:
        sysbp_target = 120.0
    diabp_target = 80.0

    # Fasting glucose
    glucose_target = 100.0

    # Total cholesterol
    totchol_target = 200.0

    # BMI range
    bmi_low, bmi_high = 18.5, 24.9

    # Sodium/day
    sodium_target = 1500.0 if (has_hypertension == 1) else 2300.0

    # Fiber/day
    fiber_target = 30.0 if (bmi is not None and bmi >= 25.0) else 25.0

    return {
        'sysbp': sysbp_target,
        'diabp': diabp_target,
        'glucose': glucose_target,
        'totchol': totchol_target,
        'bmi_low': bmi_low,
        'bmi_high': bmi_high,
        'sodium': sodium_target,
        'fiber': fiber_target,
    }


def comparison_chart_to_data_uri(*, current: dict, targets: dict, title: str = 'Your numbers vs recommended') -> str:
    """Create a grouped horizontal bar chart comparing current values vs recommended upper bounds.

    Metrics: Systolic BP, Diastolic BP, Glucose, Total Cholesterol, BMI (uses upper bound).
    """
    try:
        labels = ['Systolic BP (mmHg)', 'Diastolic BP (mmHg)', 'Glucose (mg/dL)', 'Total Chol (mg/dL)', 'BMI']
        curr_vals = [
            float(current.get('sysbp', 0) or 0),
            float(current.get('diabp', 0) or 0),
            float(current.get('glucose', 0) or 0),
            float(current.get('totchol', 0) or 0),
            float(current.get('bmi', 0) or 0),
        ]
        targ_vals = [
            float(targets.get('sysbp', 0) or 0),
            float(targets.get('diabp', 0) or 0),
            float(targets.get('glucose', 0) or 0),
            float(targets.get('totchol', 0) or 0),
            float(targets.get('bmi_high', 25.0) or 25.0),
        ]

        # Build chart
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        y_pos = np.arange(len(labels))
        ax.barh(y_pos + 0.2, targ_vals, height=0.35, color='#d5dbdb', label='Recommended (≤)')
        ax.barh(y_pos - 0.2, curr_vals, height=0.35, color='#2e86c1', label='Your value')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Value')
        ax.set_title(title)
        ax.legend(loc='lower right')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ''


