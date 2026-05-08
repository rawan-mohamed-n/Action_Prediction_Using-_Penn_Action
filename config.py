"""
config.py — Central configuration for the Penn Action pipeline.

All paths, constants, and action-specific settings live here.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT = './Penn_Action/Penn_Action'
FRAMES_DIR   = os.path.join(DATASET_ROOT, 'frames')
LABELS_DIR   = os.path.join(DATASET_ROOT, 'labels')

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
# ACTION DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS = {
    'jump_rope': {
        'keywords': ['jump_rope'],
        'range': (1, 2327),
        'interpolation': 'single',     # Single-anchor (Head only)
        'use_symmetry': True,
        'viewpoint_threshold': 0.45,
        'baseline_method': 'standing',  # Standing frame for baseline
    },
    'squat': {
        'keywords': ['squat'],
        'range': (1, 2327),
        'interpolation': 'dual',       # Dual-anchor (Head + Mid-Hip)
        'use_symmetry': True,
        'viewpoint_threshold': 0.45,
        'baseline_method': 'standing',
    },
    'pushup': {
        'keywords': ['pushup', 'pushups'],
        'range': (1, 2327),
        'interpolation': 'dual',
        'use_symmetry': True,
        'viewpoint_threshold': 0.45,
        'baseline_method': 'standing',
    },
    'pullup': {
        'keywords': ['pullup', 'pullups'],
        'range': (1, 2327),
        'interpolation': 'single',
        'use_symmetry': True,
        'viewpoint_threshold': 0.40,
        'baseline_method': 'hanging',   # Hang-distance for baseline
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FILES
# ─────────────────────────────────────────────────────────────────────────────

def joints_csv(action):    return f'{action}_joints.csv'
def angles_csv(action):    return f'{action}_angles.csv'
def summary_csv(action):   return f'{action}_normalized_summary.csv'
