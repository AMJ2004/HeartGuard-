import json
import os
import pickle
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Updated to match actual dataset columns
FEATURE_ORDER = [
    'sysBP',
    'glucose', 
    'age',
    'totChol',
    'diaBP',
    'prevalentHyp',
    'diabetes',
    'male',
    'BPMeds',
    'BMI',  # Changed from 'bmi' to 'BMI' to match dataset
]


@dataclass
class TrainResult:
    model: RandomForestClassifier
    scaler: MinMaxScaler
    threshold: float
    metrics: dict


def _load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep only necessary columns + target
    required = set(FEATURE_ORDER + ['TenYearCHD'])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df = df[list(required)].copy()
    # Drop rows with NA in required columns
    df = df.dropna(subset=list(required))
    # Ensure integer types for binary flags
    for col in ['prevalentHyp', 'diabetes', 'male', 'BPMeds', 'TenYearCHD']:
        df[col] = df[col].astype(int)
    return df


def _train(
    df: pd.DataFrame,
    random_state: int = 42,
) -> TrainResult:
    X = df[FEATURE_ORDER].values
    y = df['TenYearCHD'].values.astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)

    # Attempt SMOTE; if not available, rely on class_weight='balanced'
    use_smote = False
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore

        sm = SMOTE(random_state=random_state)
        X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)
        use_smote = True
    except Exception:
        X_train_res, y_train_res = X_train_s, y_train

    base_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        class_weight=None if use_smote else 'balanced',
    )

    # Randomized hyperparameter search to boost ROC-AUC
    param_distributions = {
        'n_estimators': [200, 300, 400, 500, 600, 800],
        'max_depth': [None, 6, 8, 10, 12, 16, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_train_res, y_train_res)
    model = search.best_estimator_

    valid_proba = model.predict_proba(X_valid_s)[:, 1]
    auc = roc_auc_score(y_valid, valid_proba)

    fpr, tpr, thr = roc_curve(y_valid, valid_proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thr[best_idx])

    y_pred = (valid_proba >= best_threshold).astype(int)
    acc = accuracy_score(y_valid, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_valid, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_valid, y_pred).tolist()

    metrics = {
        'roc_auc': float(auc),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'threshold': best_threshold,
        'confusion_matrix': cm,
        'used_smote': use_smote,
        'best_params': getattr(search, 'best_params_', None),
    }

    return TrainResult(model=model, scaler=scaler, threshold=best_threshold, metrics=metrics)


def _save_artifacts(result: TrainResult) -> None:
    os.makedirs('models', exist_ok=True)
    os.makedirs('pickle files', exist_ok=True)

    with open(os.path.join('models', 'minmax_scaler.pkl'), 'wb') as f:
        pickle.dump(result.scaler, f)

    with open(os.path.join('pickle files', 'randomf.pkl'), 'wb') as f:
        pickle.dump(result.model, f)

    with open(os.path.join('models', 'threshold.json'), 'w') as f:
        json.dump({'threshold': result.threshold}, f)

    with open(os.path.join('models', 'metrics.json'), 'w') as f:
        json.dump(result.metrics, f, indent=2)


def main():
    df = _load_data('framingham.csv')
    result = _train(df)
    _save_artifacts(result)
    print('Training complete. Metrics:')
    print(json.dumps(result.metrics, indent=2))


if __name__ == '__main__':
    main()


