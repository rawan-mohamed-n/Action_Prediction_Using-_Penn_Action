# 🏋️ Action Prediction Using Penn Action

**Camera-based exercise classification and real-time form evaluation for telerehabilitation.**

A modular Python pipeline that classifies rehabilitation exercises (Squat, Pushup, Pullup, Jump Rope) from 2D joint trajectories extracted from the [Penn Action Dataset](http://dreamdragon.github.io/PennAction/), achieving **95.14% accuracy** with an SVM classifier — plus **real-time webcam inference** with live form quality scoring.

---

## 📊 Results

### Batch Classification (PCA + SVM/RF)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM (RBF)** | **95.14%** | 0.96 | 0.95 | 0.95 |
| Random Forest | 94.44% | 0.95 | 0.94 | 0.94 |

### Form Evaluator (Real-Time Models)

| Component | Model | Metric |
|-----------|-------|--------|
| Action Classifier | Random Forest (200 trees) | **98.80% accuracy** |
| Quality Regressor | Random Forest (300 trees) | **0.006 MSE** |

### Per-Class Breakdown (SVM)

| Action | Precision | Recall | F1 | Support |
|--------|-----------|--------|-----|---------|
| Jump Rope | 1.00 | 0.94 | 0.97 | 16 |
| Pullup | 0.90 | 0.95 | 0.93 | 40 |
| Pushup | 1.00 | 0.90 | 0.95 | 42 |
| Squat | 0.94 | 1.00 | 0.97 | 46 |

---

## 🏗️ Pipeline Architecture

```
Penn Action .mat files
        │
        ▼
┌──────────────────┐
│  preprocessing.py │  Load → Mirror Symmetry → Anchor Interpolation
└────────┬─────────┘
         ▼
┌──────────────────────┐
│ feature_extraction.py │  Viewpoint Detection → Angles → Summary Stats
└────────┬─────────────┘
         ▼
┌──────────────────┐
│   training.py     │  Correlation Removal → PCA → RF + SVM
└────────┬─────────┘
         ▼
┌──────────────────┐
│  evaluation.py    │  Reports, Confusion Matrices, Feature Importance
└────────┬─────────┘
         ▼
┌──────────────────────────┐
│ form_evaluator_training.py│  Scaler + Action Classifier + Quality Regressor
└────────┬─────────────────┘
         ▼
┌──────────────────┐
│  inference.py     │  Real-time Webcam → MediaPipe → Live Predictions
└──────────────────┘
```

### Project Structure

```
├── config.py                    # Central configuration (paths, constants, feature schema)
├── helpers.py                   # Pure math utilities (distances, angles, scaling)
├── preprocessing.py             # Raw .mat → clean DataFrames
├── feature_extraction.py        # DataFrames → kinematic feature CSVs
├── training.py                  # Feature CSVs → trained RF/SVM models (batch)
├── evaluation.py                # Model evaluation and visualization
├── form_evaluator_training.py   # Train action classifier + quality regressor
├── inference.py                 # Real-time webcam inference (MediaPipe)
├── run_pipeline.py              # CLI entry point for all stages
│
├── Pushup/                      # Per-action raw CSV data
│   ├── pushup_angles.csv
│   └── pushup_joints.csv
├── Squat/
├── Jump_Rope/
├── Pullup/
│
├── output/                      # Generated outputs
│   ├── jump_rope/               #   Per-action joints, angles, summary CSVs
│   ├── squat/
│   ├── pushup/
│   ├── pullup/
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── pca_variance.png
│
├── scaler.pkl                   # Fitted StandardScaler (form evaluator)
├── action_classifier.pkl        # RF action classifier
├── quality_regressor.pkl        # RF quality regressor
│
├── form_evaluator.ipynb         # Interactive form evaluator (reference)
├── Exploration.ipynb            # Interactive EDA / visualization
└── model_training.ipynb         # Original notebook (reference only)
```

---

## 🔬 Methodology

### 1. Preprocessing

Raw Penn Action `.mat` files contain 13 joint keypoints per frame with visibility masks. Our preprocessing pipeline handles:

- **Visibility-Aware Loading**: Occluded joints (visibility=0) are set to `NaN` instead of trusting corrupted coordinates.
- **Mirror Symmetry**: For front-facing sequences, missing joints are estimated by reflecting visible contralateral joints across the head's vertical axis.
- **Anchor-Based Interpolation**: Joints are converted to coordinates relative to an anatomical anchor before interpolation, preventing "sliding" artifacts.
  - *Single-anchor* (Head) for Jump Rope and Pullup
  - *Dual-anchor* (Head + Mid-Hip) for Squat and Pushup — decouples upper/lower body motion
- **Automatic Viewpoint Detection**: Classifies camera angle as Front/Back vs Side using the shoulder-width to torso-height ratio.

### 2. Feature Extraction

- **Baseline Frame Selection**: Identifies the reference frame where limbs are at natural length (standing or hanging).
- **Dual-Method Angle Computation**:
  - *Side view*: Vector dot-product angle at each joint
  - *Front/back view*: Arcsin ratio method estimating 3D bend from 2D foreshortening
- **8 joint angles** per frame: Shoulder, Elbow, Hip, Knee (bilateral)
- **Sequence Summarization**: Each variable-length sequence → fixed-length vector via:
  - Angle statistics: max, min, mean, variance (32 features)
  - Joint movement: standard deviation of each coordinate (28 features)
- **Min-Max Normalization** to [0, 1]

### 3. Dimensionality Reduction

- Remove features with Pearson correlation > 0.90 (training set only)
- PCA retaining ≥ 95% variance
- Scaler and PCA fit only on training data (no leakage)

### 4. Batch Classification

- **Random Forest**: 200 trees, max depth 10, balanced class weights
- **SVM (RBF)**: C=1.0, γ=scale, balanced class weights
- 80/20 stratified train/test split

### 5. Form Evaluator Training

- **StandardScaler** fitted on the 60-feature kinematic vectors
- **Action Classifier**: Random Forest (200 trees) on scaled features — 98.80% accuracy
- **Quality Scoring**: Per-action IsolationForest generates anomaly-based quality scores (0.4–1.0), then a Random Forest regressor (300 trees) learns to predict them from features + action OHE

### 6. Real-Time Inference

- **MediaPipe Pose** extracts 33 landmarks per frame from webcam feed
- **Threaded camera capture** — main thread never blocks on I/O
- **Sliding window** aggregates 90 frames into the same 60-feature vector
- **Heuristic gatekeeper**: idle detection (Y-variance < threshold) skips inference
- **Asynchronous inference**: background thread runs action + quality models every N frames
- **Live HUD overlay**: action label, form quality score, FPS counter

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Full pipeline: extract → train → evaluate → form-train
python run_pipeline.py --all

# Individual stages
python run_pipeline.py --extract                 # Feature extraction only
python run_pipeline.py --extract --action squat  # Single action
python run_pipeline.py --train                   # Batch training only
python run_pipeline.py --evaluate                # Batch evaluation only
python run_pipeline.py --form-train              # Train form evaluator models
python run_pipeline.py --infer                   # Launch real-time inference
```

### Real-Time Inference

```bash
# Default webcam
python run_pipeline.py --infer

# Specific camera index or URL
python run_pipeline.py --infer --source 0
python run_pipeline.py --infer --source "http://192.168.1.100:4747/video"
```

### Dataset

Download the [Penn Action Dataset](http://dreamdragon.github.io/PennAction/) and place it at `./Penn_Action/Penn_Action/`.

---

## 👥 Team

- Ahmed Loay
- Mo'men Mohamed
- Rawan Mohamed
- Youssef Mohamed

Cairo University, Faculty of Engineering

---

## 📄 License

This project is part of the Machine Learning course at Cairo University.
