"""
training.py — Load feature CSVs → train classification models.

Handles dataset loading, feature preparation (one-hot encoding, correlation
removal, PCA), and model training (Random Forest, SVM).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from config import ACTIONS, summary_csv


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_master_dataset(actions=None):
    """
    Load all action summary CSVs and combine into a single master DataFrame
    with an 'Action_Label' column.

    Parameters:
        actions: dict of {action_name: csv_path}, or None to use all from config.
    """
    if actions is None:
        actions = {name: summary_csv(name) for name in ACTIONS}

    dataframes = []
    for action_name, csv_path in actions.items():
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping {action_name}")
            continue

        df = pd.read_csv(csv_path)
        df['Action_Label'] = action_name
        dataframes.append(df)
        print(f"  Loaded {action_name}: {len(df)} samples from {csv_path}")

    master_df = pd.concat(dataframes, ignore_index=True)
    print(f"\n  Master dataset: {len(master_df)} total samples, "
          f"{master_df['Action_Label'].nunique()} action classes")

    return master_df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(master_df, test_size=0.2, random_state=42):
    """
    Prepare features from master DataFrame:
      1. Separate X (features) and y (labels)
      2. One-hot encode categorical columns (Pose)
      3. Fill NaN, split into train/test

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = master_df.drop(columns=['Sequence_ID', 'Action_Label'])
    y = master_df['Action_Label']

    # One-hot encode the 'Pose' column
    X = pd.get_dummies(X, columns=['Pose'], drop_first=False).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"  Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def remove_correlated_features(X_train, X_test, threshold=0.90):
    """
    Remove features with correlation > threshold (computed on training set only).
    Returns reduced X_train, X_test, and list of dropped columns.
    """
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    X_train_reduced = X_train.drop(columns=to_drop)
    X_test_reduced  = X_test.drop(columns=to_drop)

    print(f"  Dropped {len(to_drop)} correlated features "
          f"(>{threshold}): {X_train.shape[1]} → {X_train_reduced.shape[1]}")

    return X_train_reduced, X_test_reduced, to_drop


def apply_pca(X_train, X_test, variance_ratio=0.95, random_state=42):
    """
    Scale features and apply PCA to retain specified variance ratio.
    Returns transformed arrays, fitted scaler, and fitted PCA.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=variance_ratio, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    print(f"  PCA: {X_train.shape[1]} features → {pca.n_components_} components "
          f"(retaining {variance_ratio*100:.0f}% variance)")

    return X_train_pca, X_test_pca, scaler, pca


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, n_estimators=200, max_depth=10,
                        random_state=42):
    """Train a Random Forest classifier."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    print(f"  Random Forest trained ({n_estimators} trees, max_depth={max_depth})")
    return clf


def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale',
              random_state=42):
    """Train an SVM classifier."""
    model = SVC(
        kernel=kernel, C=C, gamma=gamma,
        class_weight='balanced', random_state=random_state
    )
    model.fit(X_train, y_train)
    print(f"  SVM trained (kernel={kernel}, C={C})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline():
    """
    Full training pipeline:
      1. Load master dataset
      2. Prepare features
      3. Remove correlated features
      4. Apply PCA
      5. Train Random Forest and SVM
      6. Return models and test data for evaluation
    """
    print("\n" + "="*60)
    print("  TRAINING PIPELINE")
    print("="*60)

    # 1. Load
    print("\n[1/5] Loading dataset...")
    master_df = load_master_dataset()

    # 2. Prepare
    print("\n[2/5] Preparing features...")
    X_train, X_test, y_train, y_test = prepare_features(master_df)

    # 3. Remove correlated
    print("\n[3/5] Removing correlated features...")
    X_train_r, X_test_r, dropped = remove_correlated_features(X_train, X_test)

    # 4. PCA
    print("\n[4/5] Applying PCA...")
    X_train_pca, X_test_pca, scaler, pca = apply_pca(X_train_r, X_test_r)

    # 5. Train models
    print("\n[5/5] Training models...")
    rf_model  = train_random_forest(X_train_pca, y_train)
    svm_model = train_svm(X_train_pca, y_train)

    return {
        'master_df': master_df,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_pca': X_train_pca, 'X_test_pca': X_test_pca,
        'scaler': scaler, 'pca': pca,
        'rf_model': rf_model,
        'svm_model': svm_model,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = run_training_pipeline()
    print("\nTraining complete. Use evaluation.py to see results.")
