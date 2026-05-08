"""
form_evaluator_training.py — Train action classifier + quality regressor.

Converts the logic from form_evaluator.ipynb into a reproducible module.
Reads per-action joints/angles CSVs, computes summary features, fits:
  1. StandardScaler (→ scaler.pkl)
  2. RandomForest action classifier (→ action_classifier.pkl)
  3. IsolationForest quality scores + RandomForest regressor (→ quality_regressor.pkl)

All constants are imported from config.py.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import (
    ANGLE_KEYS,
    FEATURE_COLS,
    LANDMARK_KEYS,
    RAW_CSV_PATHS,
    SCALER_PATH,
    ACTION_MODEL_PATH,
    QUALITY_MODEL_PATH,
    ACTION_OHE_COLS,
    ACTION_LABELS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SUMMARISATION
# ─────────────────────────────────────────────────────────────────────────────

def summarize_angles(df_angles: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-frame angles into per-sequence summary statistics."""
    df = df_angles[['Sequence_ID'] + ANGLE_KEYS].copy()
    for col in ANGLE_KEYS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    grouped = df.groupby('Sequence_ID', as_index=True)
    parts = []
    for col in ANGLE_KEYS:
        g = grouped[col]
        parts.append(pd.DataFrame({
            f'{col}_max': g.max(),
            f'{col}_min': g.min(),
            f'{col}_avg': g.mean(),
            f'{col}_var': g.var(ddof=0),
        }))
    return pd.concat(parts, axis=1)


def summarize_joints(df_joints: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-frame joint coordinates into per-sequence std."""
    df = df_joints.copy()

    # Compute Mid_Hip if missing
    if 'Mid_Hip_x' not in df.columns and {'L_Hip_x', 'R_Hip_x'}.issubset(df.columns):
        df['Mid_Hip_x'] = (
            pd.to_numeric(df['L_Hip_x'], errors='coerce')
            + pd.to_numeric(df['R_Hip_x'], errors='coerce')
        ) / 2
    if 'Mid_Hip_y' not in df.columns and {'L_Hip_y', 'R_Hip_y'}.issubset(df.columns):
        df['Mid_Hip_y'] = (
            pd.to_numeric(df['L_Hip_y'], errors='coerce')
            + pd.to_numeric(df['R_Hip_y'], errors='coerce')
        ) / 2

    for lm in LANDMARK_KEYS:
        for axis in ('x', 'y'):
            col = f'{lm}_{axis}'
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

    coord_cols = [f'{lm}_x' for lm in LANDMARK_KEYS] + [f'{lm}_y' for lm in LANDMARK_KEYS]
    summary = df.groupby('Sequence_ID', as_index=True)[coord_cols].std(ddof=0)
    summary = summary.rename(columns={c: f'{c}_std' for c in coord_cols})
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_action_features(label: str, angles_path, joints_path) -> pd.DataFrame:
    """Load and merge angle/joint summaries for a single action."""
    angles = pd.read_csv(angles_path)
    joints = pd.read_csv(joints_path)

    angle_summary = summarize_angles(angles)
    joint_summary = summarize_joints(joints)

    merged = angle_summary.join(joint_summary, how='inner')
    merged['action_label'] = label
    return merged.reset_index()


def build_form_dataset():
    """
    Build the full dataset from all per-action CSVs.

    Returns:
        df:            Master DataFrame (Sequence_ID, action_label, 60 features)
        X_scaled:      Scaled feature matrix (DataFrame)
        y:             Action labels (Series)
        scaler:        Fitted StandardScaler
        imputer:       Fitted SimpleImputer
    """
    frames = []
    for label, files in RAW_CSV_PATHS.items():
        if not files['angles'].exists() or not files['joints'].exists():
            logger.warning('Missing CSV files for %s, skipping.', label)
            continue
        frames.append(load_action_features(label, files['angles'], files['joints']))
        logger.info('Loaded %s: %d sequences', label, len(frames[-1]))

    df = pd.concat(frames, ignore_index=True)

    # Ensure all 60 feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df.reindex(columns=['Sequence_ID', 'action_label'] + FEATURE_COLS)

    X_raw = df[FEATURE_COLS]
    y = df['action_label']

    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=FEATURE_COLS)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=FEATURE_COLS)

    return df, X_scaled, y, scaler, imputer


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_action_classifier(X_scaled, y):
    """Train and evaluate a RandomForest action classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n  Action classifier accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred))

    return clf


def train_quality_regressor(X_scaled, y, df):
    """
    Generate quality scores via per-action IsolationForest, then train
    a RandomForest regressor to predict them.
    """
    quality_scores = np.zeros(len(df), dtype=float)

    for label in df['action_label'].unique():
        idx = df['action_label'] == label
        X_group = X_scaled.loc[idx]

        iso = IsolationForest(
            n_estimators=200, contamination='auto',
            random_state=42, n_jobs=-1,
        )
        iso.fit(X_group)

        scores = iso.decision_function(X_group).reshape(-1, 1)
        score_scaler = MinMaxScaler(feature_range=(0.4, 1.0))
        scaled_scores = score_scaler.fit_transform(scores).ravel()
        quality_scores[idx.to_numpy()] = scaled_scores

    df['quality_score'] = quality_scores

    # Build regressor input: features + action OHE
    action_ohe = pd.get_dummies(df['action_label'], prefix='action')
    action_ohe = action_ohe.reindex(columns=ACTION_OHE_COLS, fill_value=0)

    X_quality = pd.concat(
        [X_scaled.reset_index(drop=True), action_ohe.reset_index(drop=True)],
        axis=1,
    )
    y_quality = df['quality_score'].values

    Xq_train, Xq_test, yq_train, yq_test = train_test_split(
        X_quality, y_quality, test_size=0.2, random_state=42,
    )

    reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    reg.fit(Xq_train, yq_train)

    yq_pred = reg.predict(Xq_test)
    mse = mean_squared_error(yq_test, yq_pred)
    print(f'  Quality regressor MSE: {mse:.6f}\n')

    return reg


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_form_evaluator_pipeline():
    """
    End-to-end: load data → train scaler + action classifier + quality regressor.
    Saves three .pkl files to the project root.
    """
    print('\n' + '=' * 60)
    print('  FORM EVALUATOR TRAINING')
    print('=' * 60)

    print('\n[1/4] Building dataset...')
    df, X_scaled, y, scaler, _imputer = build_form_dataset()
    print(f'  Dataset: {len(df)} samples, {df["action_label"].nunique()} actions')

    print('\n[2/4] Saving scaler...')
    joblib.dump(scaler, SCALER_PATH)
    print(f'  Saved: {SCALER_PATH}')

    print('\n[3/4] Training action classifier...')
    action_clf = train_action_classifier(X_scaled, y)
    joblib.dump(action_clf, ACTION_MODEL_PATH)
    print(f'  Saved: {ACTION_MODEL_PATH}')

    print('\n[4/4] Training quality regressor...')
    quality_reg = train_quality_regressor(X_scaled, y, df)
    joblib.dump(quality_reg, QUALITY_MODEL_PATH)
    print(f'  Saved: {QUALITY_MODEL_PATH}')

    print('\nForm evaluator training complete.')
    return {
        'scaler': scaler,
        'action_classifier': action_clf,
        'quality_regressor': quality_reg,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    run_form_evaluator_pipeline()
