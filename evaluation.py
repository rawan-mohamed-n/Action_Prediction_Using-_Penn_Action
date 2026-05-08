"""
evaluation.py — Model evaluation utilities.

Classification reports, confusion matrices, feature importance,
and correlation matrix visualization. Each pattern that was
copy-pasted for RF and SVM in the notebook is now a single function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


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
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(model, y_test, y_pred, model_name="Model",
                          cmap='Blues', figsize=(8, 6)):
    """
    Plot a confusion matrix heatmap.
    This was copy-pasted identically for RF and SVM in the notebook.
    """
    labels = sorted(y_test.unique()) if hasattr(y_test, 'unique') else model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.title(f'{model_name} — Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, top_n=15):
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
        plt.savefig(save_path, dpi=300)
        print(f"  Saved correlation matrix to {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PCA VARIANCE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_variance(pca, target_ratio=0.95):
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
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(results):
    """
    Run full evaluation on training results dict (from training.run_training_pipeline).
    """
    X_test_pca = results['X_test_pca']
    y_test     = results['y_test']
    rf_model   = results['rf_model']
    svm_model  = results['svm_model']
    pca        = results['pca']
    master_df  = results['master_df']

    print("\n" + "="*60)
    print("  EVALUATION")
    print("="*60)

    # Random Forest
    y_pred_rf, acc_rf = evaluate_model(rf_model, X_test_pca, y_test, "Random Forest")
    plot_confusion_matrix(rf_model, y_test, y_pred_rf, "Random Forest", cmap='Blues')

    # SVM
    y_pred_svm, acc_svm = evaluate_model(svm_model, X_test_pca, y_test, "SVM")
    plot_confusion_matrix(svm_model, y_test, y_pred_svm, "SVM", cmap='Oranges')

    # Feature importance (RF only)
    if results.get('X_train') is not None:
        plot_feature_importance(rf_model, results['X_train'].columns)

    # PCA variance
    plot_pca_variance(pca)

    # Correlation matrix
    plot_correlation_matrix(master_df, save_path='kinematic_correlation_matrix.png')

    return {
        'rf_accuracy': acc_rf,
        'svm_accuracy': acc_svm,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Run this via run_pipeline.py or import and call run_evaluation(results)")
