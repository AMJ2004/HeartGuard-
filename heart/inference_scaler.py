from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURE_ORDER = ['sysBP', 'glucose', 'age', 'totChol', 'diaBP', 'prevalentHyp', 'diabetes', 'male', 'BPMeds', 'BMI']


def _fit_scaler_from_framingham(csv_path: str) -> MinMaxScaler:
    data = pd.read_csv(csv_path)
    # Minimal cleaning consistent with training
    data["glucose"].fillna((data["glucose"].mode())[0], inplace=True)
    data.dropna(inplace=True)

    # Feature subset consistent with app's input
    # BMI may be named 'BMI' in dataset; create lowercase 'bmi'
    if 'BMI' in data.columns and 'bmi' not in data.columns:
        data['bmi'] = data['BMI']

    cols = ['sysBP', 'glucose', 'age', 'totChol', 'diaBP', 'prevalentHyp', 'diabetes', 'male', 'BPMeds', 'bmi']
    X = data[cols].copy()

    scaler = MinMaxScaler()
    scaler.fit(X.values)
    return scaler


def get_minmax_scaler(*, model_dir: str = 'models', csv_fallback: Optional[str] = 'framingham.csv') -> Tuple[MinMaxScaler, str]:
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, 'minmax_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            return scaler, scaler_path

    if csv_fallback and os.path.exists(csv_fallback):
        scaler = _fit_scaler_from_framingham(csv_fallback)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        return scaler, scaler_path

    # Fallback to identity scaler if nothing available (avoids crash, but may reduce accuracy)
    scaler = MinMaxScaler()
    scaler.fit(np.zeros((1, len(FEATURE_ORDER))))
    return scaler, scaler_path


