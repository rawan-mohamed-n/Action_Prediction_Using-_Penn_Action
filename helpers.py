"""
helpers.py — Low-level math utilities for kinematic computations.

These are pure functions with no side effects or I/O.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE & ANGLE COMPUTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calc_distance(row, j1, j2):
    """Calculates 2D Euclidean distance between two joints in a DataFrame row."""
    if pd.isna(row[f'{j1}_x']) or pd.isna(row[f'{j2}_x']):
        return np.nan
    return np.sqrt((row[f'{j1}_x'] - row[f'{j2}_x'])**2 +
                   (row[f'{j1}_y'] - row[f'{j2}_y'])**2)


def calc_vector_angle(row, j1, j2, j3):
    """
    Calculates the angle at joint j2 formed by vectors j2→j1 and j2→j3
    using the dot product. Used for side-view poses.
    """
    keys = [f'{j1}_x', f'{j1}_y', f'{j2}_x', f'{j2}_y', f'{j3}_x', f'{j3}_y']
    if any(pd.isna(row[k]) for k in keys):
        return np.nan

    v1 = np.array([row[f'{j1}_x'] - row[f'{j2}_x'],
                   row[f'{j1}_y'] - row[f'{j2}_y']])
    v2 = np.array([row[f'{j3}_x'] - row[f'{j2}_x'],
                   row[f'{j3}_y'] - row[f'{j2}_y']])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def calc_arcsin_angle(d_current_1, d_baseline_1, d_current_2, d_baseline_2):
    """
    Estimates joint bend angle from projected limb-length ratios using arcsin.
    Used for front/back-view poses.
    """
    if any(pd.isna(v) for v in [d_current_1, d_baseline_1, d_current_2, d_baseline_2]):
        return np.nan
    if d_baseline_1 == 0 or d_baseline_2 == 0:
        return np.nan

    ratio1 = np.clip(d_current_1 / d_baseline_1, 0, 1)
    ratio2 = np.clip(d_current_2 / d_baseline_2, 0, 1)

    angle1 = np.degrees(np.arcsin(ratio1))
    angle2 = np.degrees(np.arcsin(ratio2))
    return abs(angle1 - angle2)


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE & ANGLE COMPUTATION (HIGHER-LEVEL)
# ─────────────────────────────────────────────────────────────────────────────

def compute_baselines(stand_row):
    """
    Compute baseline limb distances from a reference (standing/hanging) frame.
    Returns a dict of 10 baseline segment lengths.
    """
    return {
        'L_SH': calc_distance(stand_row, 'L_Sho', 'L_Hip'),
        'L_HK': calc_distance(stand_row, 'L_Hip', 'L_Kne'),
        'L_KA': calc_distance(stand_row, 'L_Kne', 'L_Ank'),
        'L_SE': calc_distance(stand_row, 'L_Sho', 'L_Elb'),
        'L_EW': calc_distance(stand_row, 'L_Elb', 'L_Wri'),
        'R_SH': calc_distance(stand_row, 'R_Sho', 'R_Hip'),
        'R_HK': calc_distance(stand_row, 'R_Hip', 'R_Kne'),
        'R_KA': calc_distance(stand_row, 'R_Kne', 'R_Ank'),
        'R_SE': calc_distance(stand_row, 'R_Sho', 'R_Elb'),
        'R_EW': calc_distance(stand_row, 'R_Elb', 'R_Wri'),
    }


# Joint-angle definitions: (parent, joint, child, baseline_key1, baseline_key2)
ANGLE_DEFS = {
    'L_Shoulder_Angle': ('L_Hip', 'L_Sho', 'L_Elb', 'L_SH', 'L_SE'),
    'R_Shoulder_Angle': ('R_Hip', 'R_Sho', 'R_Elb', 'R_SH', 'R_SE'),
    'L_Elbow_Angle':    ('L_Sho', 'L_Elb', 'L_Wri', 'L_SE', 'L_EW'),
    'R_Elbow_Angle':    ('R_Sho', 'R_Elb', 'R_Wri', 'R_SE', 'R_EW'),
    'L_Hip_Angle':      ('L_Sho', 'L_Hip', 'L_Kne', 'L_SH', 'L_HK'),
    'R_Hip_Angle':      ('R_Sho', 'R_Hip', 'R_Kne', 'R_SH', 'R_HK'),
    'L_Knee_Angle':     ('L_Hip', 'L_Kne', 'L_Ank', 'L_HK', 'L_KA'),
    'R_Knee_Angle':     ('R_Hip', 'R_Kne', 'R_Ank', 'R_HK', 'R_KA'),
}


def compute_frame_angles(row, sequence_id, pose_type, base):
    """
    Compute all 8 kinematic joint angles for a single frame.
    Returns a dict with Sequence_ID, Frame, Pose, and 8 angle values.
    """
    ang = {'Sequence_ID': sequence_id, 'Frame': row['Frame'], 'Pose': pose_type}

    for angle_name, (j1, j2, j3, bk1, bk2) in ANGLE_DEFS.items():
        if pose_type == 'Front/Back':
            ang[angle_name] = calc_arcsin_angle(
                calc_distance(row, j1, j2), base[bk1],
                calc_distance(row, j2, j3), base[bk2]
            )
        else:
            ang[angle_name] = calc_vector_angle(row, j1, j2, j3)

    # Round angles
    for k in ang:
        if 'Angle' in k and not pd.isna(ang[k]):
            ang[k] = round(ang[k], 2)

    return ang


def compute_all_angles(df, sequence_id, pose_type, base):
    """Compute kinematic angles for every frame in a DataFrame."""
    return [compute_frame_angles(row, sequence_id, pose_type, base)
            for _, row in df.iterrows()]


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def min_max_scale(column):
    """Min-Max normalize a single pandas Series to [0, 1]."""
    return (column - column.min()) / (column.max() - column.min() + 1e-8)
