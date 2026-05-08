"""
evaluation.py — Model evaluation utilities.

Classification reports, confusion matrices, feature importance,
and correlation matrix visualization. Each pattern that was
copy-pasted for RF and SVM in the notebook is now a single function.

All plots are saved to the output/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from config import OUTPUT_DIR, ensure_output_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Print classification report and accuracy for a trained model.
    Returns (y_pred, accuracy).
    """
    y_pred = model.predict(X_test)

    print(f"\n{'─'*50}")
    print(f"  {model_name} — Classification Report")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print(f"  Overall Accuracy: {acc * 100:.2f}%\n")

    return y_pred, acc


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRICES (SIDE BY SIDE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(y_test, y_pred_rf, y_pred_svm, labels,
                            save_path=None):
    """
    Plot both model confusion matrices side by side in a single figure.
    """
    cm_rf  = confusion_matrix(y_test, y_pred_rf,  labels=labels)
    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Random Forest — Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('SVM — Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved confusion matrices to {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    """
    Plot top-N most important features from a tree-based model.
    """
    if not hasattr(model, 'feature_importances_'):
        print("  Model does not support feature_importances_.")
        return

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved feature importance to {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_matrix(df, save_path=None, figsize=(24, 20)):
    """
    Plot a correlation heatmap for numerical features in a DataFrame.
    """
    cols_to_drop = ['Sequence_ID', 'Pose', 'Action_Label']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    numeric_df = df.drop(columns=cols_to_drop).fillna(0)

    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix, mask=mask, cmap='coolwarm',
        vmax=1.0, vmin=-1.0, center=0, square=True,
        linewidths=.5, cbar_kws={"shrink": .75}, annot=False
    )
    plt.title('Kinematic Feature Correlation Matrix', fontsize=24, pad=20)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved correlation matrix to {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PCA VARIANCE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_variance(pca, target_ratio=0.95, save_path=None):
    """Plot cumulative explained variance for PCA components."""
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Components Needed to Reach Target Variance')
    plt.axhline(y=target_ratio, color='r', linestyle='-',
                label=f'{target_ratio*100:.0f}% threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved PCA variance plot to {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(results):
    """
    Run full evaluation on training results dict (from training.run_training_pipeline).
    All plots are saved to the output/ directory.
    """
    ensure_output_dir()

    X_test_pca = results['X_test_pca']
    y_test     = results['y_test']
    rf_model   = results['rf_model']
    svm_model  = results['svm_model']
    pca        = results['pca']
    master_df  = results['master_df']

    print("\n" + "="*60)
    print("  EVALUATION")
    print("="*60)

    # Classification reports
    y_pred_rf,  acc_rf  = evaluate_model(rf_model,  X_test_pca, y_test, "Random Forest")
    y_pred_svm, acc_svm = evaluate_model(svm_model, X_test_pca, y_test, "SVM")

    # Combined confusion matrices (single window)
    labels = sorted(y_test.unique())
    plot_confusion_matrices(
        y_test, y_pred_rf, y_pred_svm, labels,
        save_path=os.path.join(OUTPUT_DIR, 'confusion_matrices.png')
    )

    # Feature importance (RF only)
    if results.get('X_train') is not None:
        plot_feature_importance(
            rf_model, results['X_train'].columns,
            save_path=os.path.join(OUTPUT_DIR, 'feature_importance.png')
        )

    # PCA variance
    plot_pca_variance(
        pca,
        save_path=os.path.join(OUTPUT_DIR, 'pca_variance.png')
    )

    # Correlation matrix
    plot_correlation_matrix(
        master_df,
        save_path=os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
    )

    return {
        'rf_accuracy': acc_rf,
        'svm_accuracy': acc_svm,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Run this via run_pipeline.py or import and call run_evaluation(results)")
