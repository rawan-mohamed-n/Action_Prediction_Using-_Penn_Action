"""
config.py — Central configuration for the Penn Action pipeline.

All paths, constants, and action-specific settings live here.
This is the single source of truth for both the batch training
pipeline and the real-time inference system.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent
DATASET_ROOT  = './Penn_Action/Penn_Action'
FRAMES_DIR    = os.path.join(DATASET_ROOT, 'frames')
LABELS_DIR    = os.path.join(DATASET_ROOT, 'labels')
OUTPUT_DIR    = './output'

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARTIFACT PATHS  (produced by form_evaluator_training.py)
# ─────────────────────────────────────────────────────────────────────────────

SCALER_PATH        = PROJECT_ROOT / 'scaler.pkl'
ACTION_MODEL_PATH  = PROJECT_ROOT / 'action_classifier.pkl'
QUALITY_MODEL_PATH = PROJECT_ROOT / 'quality_regressor.pkl'

# ─────────────────────────────────────────────────────────────────────────────
# PER-ACTION RAW CSV DIRECTORIES
# (form_evaluator_training reads from these)
# ─────────────────────────────────────────────────────────────────────────────

RAW_CSV_PATHS = {
    'pushup': {
        'angles': PROJECT_ROOT / 'Pushup'  / 'pushup_angles.csv',
        'joints': PROJECT_ROOT / 'Pushup'  / 'pushup_joints.csv',
    },
    'squat': {
        'angles': PROJECT_ROOT / 'Squat'   / 'squat_angles.csv',
        'joints': PROJECT_ROOT / 'Squat'   / 'squat_joints.csv',
    },
    'jump_rope': {
        'angles': PROJECT_ROOT / 'Jump_Rope' / 'jump_rope_angles.csv',
        'joints': PROJECT_ROOT / 'Jump_Rope' / 'jump_rope_joints.csv',
    },
    'pullup': {
        'angles': PROJECT_ROOT / 'Pullup'  / 'pullup_angles.csv',
        'joints': PROJECT_ROOT / 'Pullup'  / 'pullup_joints.csv',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# JOINT / SKELETON CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

POSE_NAMES = [
    'Head', 'L_Sho', 'R_Sho', 'L_Elb', 'R_Elb', 'L_Wri', 'R_Wri',
    'L_Hip', 'R_Hip', 'L_Kne', 'R_Kne', 'L_Ank', 'R_Ank'
]

SKELETON = [
    (0, 1), (0, 2),     # Head → Shoulders
    (1, 3), (3, 5),     # L_Sho → L_Elb → L_Wri
    (2, 4), (4, 6),     # R_Sho → R_Elb → R_Wri
    (1, 7), (2, 8),     # Shoulders → Hips
    (7, 9), (9, 11),    # L_Hip → L_Kne → L_Ank
    (8, 10), (10, 12)   # R_Hip → R_Kne → R_Ank
]

ANGLE_COLUMNS = [
    'Sequence_ID', 'Frame', 'Pose',
    'L_Shoulder_Angle', 'R_Shoulder_Angle',
    'L_Elbow_Angle', 'R_Elbow_Angle',
    'L_Hip_Angle', 'R_Hip_Angle',
    'L_Knee_Angle', 'R_Knee_Angle'
]

UPPER_JOINTS = ['L_Sho', 'R_Sho', 'L_Elb', 'R_Elb', 'L_Wri', 'R_Wri']
LOWER_JOINTS = ['L_Hip', 'R_Hip', 'L_Kne', 'R_Kne', 'L_Ank', 'R_Ank']

DEFAULT_POSE_MAP = {
    'Side': 'Side',
    'Left/Right': 'Side',
    'Front/Back': 'Front'
}

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE FEATURE SCHEMA
# Shared between form_evaluator_training.py and inference.py
# ─────────────────────────────────────────────────────────────────────────────

# 8 angle keys used in both batch feature extraction and real-time inference
ANGLE_KEYS = [
    'L_Shoulder_Angle', 'R_Shoulder_Angle',
    'L_Elbow_Angle',    'R_Elbow_Angle',
    'L_Hip_Angle',      'R_Hip_Angle',
    'L_Knee_Angle',     'R_Knee_Angle',
]

# 14 landmark keys for coordinate tracking (includes Mid_Hip)
LANDMARK_KEYS = [
    'Head', 'L_Sho', 'R_Sho',
    'L_Elb', 'R_Elb', 'L_Wri', 'R_Wri',
    'L_Hip', 'R_Hip', 'L_Kne', 'R_Kne',
    'L_Ank', 'R_Ank', 'Mid_Hip',
]

# 60-column feature vector  (8 angles × 4 stats + 14 landmarks × 2 axes)
FEATURE_COLS: list[str] = []
for _angle in ANGLE_KEYS:
    FEATURE_COLS.extend(
        [f'{_angle}_max', f'{_angle}_min', f'{_angle}_avg', f'{_angle}_var']
    )
for _lm in LANDMARK_KEYS:
    FEATURE_COLS.extend([f'{_lm}_x_std', f'{_lm}_y_std'])

# Action labels (alphabetically sorted for consistent OHE)
ACTION_LABELS: list[str] = sorted(['jump_rope', 'pullup', 'pushup', 'squat'])
ACTION_OHE_COLS: list[str] = [f'action_{a}' for a in ACTION_LABELS]

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE RUNTIME CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

IDLE_Y_KEYS: list[str]       = ['L_Hip_y', 'R_Hip_y', 'L_Sho_y', 'R_Sho_y']
IDLE_VARIANCE_THRESHOLD      = 0.001
IDLE_WINDOW_SIZE             = 5       # buffered frames checked for idleness
INFERENCE_EVERY_N_FRAMES     = 5
ACTION_WINDOW_SIZE           = 30      # Stage 1: fast, recency-focused
MAIN_BUFFER_MAXLEN           = 90      # Stage 2: deep quality window

# ─────────────────────────────────────────────────────────────────────────────
# ACTION DEFINITIONS  (batch pipeline)
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS = {
    'jump_rope': {
        'keywords': ['jump_rope'],
        'range': (955, 1036),
        'interpolation': 'single',     # Single-anchor (Head only)
        'use_symmetry': True,
        'viewpoint_threshold': 0.45,
        'baseline_method': 'standing',  # Standing frame for baseline
    },
    'squat': {
        'keywords': ['squat'],
        'range': (1659, 1889),
        'interpolation': 'dual',       # Dual-anchor (Head + Mid-Hip)
        'use_symmetry': True,
        'viewpoint_threshold': 0.45,
        'baseline_method': 'standing',
    },
    'pushup': {
        'keywords': ['pushup', 'pushups'],
        'range': (1348, 1557),
        'interpolation': 'dual',
        'use_symmetry': True,
        'viewpoint_threshold': 0.45,
        'baseline_method': 'standing',
    },
    'pullup': {
        'keywords': ['pullup', 'pullups'],
        'range': (1149, 1347),
        'interpolation': 'single',
        'use_symmetry': True,
        'viewpoint_threshold': 0.40,
        'baseline_method': 'hanging',   # Hang-distance for baseline
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FILES
# ─────────────────────────────────────────────────────────────────────────────

def action_output_dir(action):
    """Return the output directory for a specific action, creating it if needed."""
    path = os.path.join(OUTPUT_DIR, action)
    os.makedirs(path, exist_ok=True)
    return path

def ensure_output_dir():
    """Ensure the root output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def joints_csv(action):    return os.path.join(action_output_dir(action), f'{action}_joints.csv')
def angles_csv(action):    return os.path.join(action_output_dir(action), f'{action}_angles.csv')
def summary_csv(action):   return os.path.join(action_output_dir(action), f'{action}_normalized_summary.csv')
