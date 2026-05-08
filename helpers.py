"""
helpers.py — Shared constants and utility functions for Penn Action exploration.

This module centralizes all repeated code from the Exploration notebook so that
each function is defined exactly once and simply called where needed.
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

POSE_NAMES = [
    'Head', 'L_Sho', 'R_Sho', 'L_Elb', 'R_Elb', 'L_Wri', 'R_Wri',
    'L_Hip', 'R_Hip', 'L_Kne', 'R_Kne', 'L_Ank', 'R_Ank'
]

SKELETON = [
    (0, 1), (0, 2),   # Head → Shoulders
    (1, 3), (3, 5),   # L_Sho → L_Elb → L_Wri
    (2, 4), (4, 6),   # R_Sho → R_Elb → R_Wri
    (1, 7), (2, 8),   # Shoulders → Hips
    (7, 9), (9, 11),  # L_Hip → L_Kne → L_Ank
    (8, 10), (10, 12) # R_Hip → R_Kne → R_Ank
]

ANGLE_COLUMNS = [
    'Sequence_ID', 'Frame', 'Pose',
    'L_Shoulder_Angle', 'R_Shoulder_Angle',
    'L_Elbow_Angle', 'R_Elbow_Angle',
    'L_Hip_Angle', 'R_Hip_Angle',
    'L_Knee_Angle', 'R_Knee_Angle'
]

DEFAULT_POSE_MAP = {
    'Side': 'Side',
    'Left/Right': 'Side',
    'Front/Back': 'Front'
}


# ─────────────────────────────────────────────────────────────────────────────
# KINEMATIC HELPER FUNCTIONS
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
# BASELINE & ANGLE COMPUTATION
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


def compute_frame_angles(row, sequence_id, pose_type, base):
    """
    Compute all 8 kinematic joint angles for a single frame.
    Selects front/back (arcsin) or side (vector) method based on pose_type.
    Returns a dict with Sequence_ID, Frame, Pose, and 8 angle values.
    """
    ang = {'Sequence_ID': sequence_id, 'Frame': row['Frame'], 'Pose': pose_type}

    # Define joint triplets: (parent, joint, child) for each angle
    angle_defs = {
        'L_Shoulder_Angle': ('L_Hip', 'L_Sho', 'L_Elb', 'L_SH', 'L_SE'),
        'R_Shoulder_Angle': ('R_Hip', 'R_Sho', 'R_Elb', 'R_SH', 'R_SE'),
        'L_Elbow_Angle':    ('L_Sho', 'L_Elb', 'L_Wri', 'L_SE', 'L_EW'),
        'R_Elbow_Angle':    ('R_Sho', 'R_Elb', 'R_Wri', 'R_SE', 'R_EW'),
        'L_Hip_Angle':      ('L_Sho', 'L_Hip', 'L_Kne', 'L_SH', 'L_HK'),
        'R_Hip_Angle':      ('R_Sho', 'R_Hip', 'R_Kne', 'R_SH', 'R_HK'),
        'L_Knee_Angle':     ('L_Hip', 'L_Kne', 'L_Ank', 'L_HK', 'L_KA'),
        'R_Knee_Angle':     ('R_Hip', 'R_Kne', 'R_Ank', 'R_HK', 'R_KA'),
    }

    for angle_name, (j1, j2, j3, bk1, bk2) in angle_defs.items():
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
    """
    Compute kinematic angles for every frame in a DataFrame.
    Returns a list of angle dicts (one per frame).
    """
    angles = []
    for _, row in df.iterrows():
        angles.append(compute_frame_angles(row, sequence_id, pose_type, base))
    return angles


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

def anchor_interpolate(df, pose_names=None):
    """
    Anchor-based relative interpolation: interpolate all joints relative to the
    Head anchor, then reconstruct absolute coordinates.
    """
    if pose_names is None:
        pose_names = POSE_NAMES

    df['Head_x'] = df['Head_x'].interpolate(limit_direction='both')
    df['Head_y'] = df['Head_y'].interpolate(limit_direction='both')

    for name in pose_names[1:]:
        df[f'{name}_rel_x'] = df[f'{name}_x'] - df['Head_x']
        df[f'{name}_rel_y'] = df[f'{name}_y'] - df['Head_y']

        df[f'{name}_rel_x'] = df[f'{name}_rel_x'].interpolate(limit_direction='both')
        df[f'{name}_rel_y'] = df[f'{name}_rel_y'].interpolate(limit_direction='both')

        df[f'{name}_x'] = df[f'{name}_rel_x'] + df['Head_x']
        df[f'{name}_y'] = df[f'{name}_rel_y'] + df['Head_y']

        df.drop(columns=[f'{name}_rel_x', f'{name}_rel_y'], inplace=True, errors='ignore')

    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_joints_and_angles(all_joints_data, all_angles_data, action_name):
    """
    Export joint and angle DataFrames to CSV files.
    Files: {action_name}_joints.csv, {action_name}_angles.csv
    """
    if all_joints_data:
        pd.concat(all_joints_data, ignore_index=True).to_csv(
            f'{action_name}_joints.csv', index=False)

        pd.DataFrame(all_angles_data).reindex(columns=ANGLE_COLUMNS).to_csv(
            f'{action_name}_angles.csv', index=False)

        print(f"\nSuccess! Files saved: {action_name}_joints.csv, {action_name}_angles.csv")
    else:
        print(f"No {action_name} sequences found.")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZE & NORMALIZE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def min_max_scale(column):
    """Min-Max normalize a single pandas Series to [0, 1]."""
    return (column - column.min()) / (column.max() - column.min() + 1e-8)


def summarize_and_normalize(action_name, pose_map=None):
    """
    Full pipeline: load joints/angles CSVs → compute summary statistics →
    normalize → export as {action_name}_normalized_summary.csv.

    Steps:
      1. Load {action_name}_joints.csv and {action_name}_angles.csv
      2. Group angles by (Sequence_ID, Pose) → agg max/min/mean/var
      3. Group joints by Sequence_ID → agg std
      4. Flatten multi-index columns
      5. Merge, rename mean→avg, map poses, min-max normalize
      6. Export
    """
    if pose_map is None:
        pose_map = DEFAULT_POSE_MAP

    # 1. Load data
    df_joints = pd.read_csv(f'{action_name}_joints.csv')
    df_angles = pd.read_csv(f'{action_name}_angles.csv')

    # 2. Process Angles (Max, Min, Avg, Variance)
    angle_cols = [col for col in df_angles.columns if 'Angle' in col]
    angle_summary = df_angles.groupby(['Sequence_ID', 'Pose'])[angle_cols].agg(
        ['max', 'min', 'mean', 'var']
    ).reset_index()

    # Flatten multi-index columns
    angle_summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in angle_summary.columns.values
    ]

    # 3. Process Joint Distributions (std of movement)
    joint_cols = [col for col in df_joints.columns if '_x' in col or '_y' in col]
    joint_summary = df_joints.groupby('Sequence_ID')[joint_cols].agg(
        ['std']
    ).reset_index()

    # Flatten joint columns
    joint_summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in joint_summary.columns.values
    ]

    # 4. Merge and clean up
    final_df = pd.merge(angle_summary, joint_summary, on='Sequence_ID', how='inner')
    final_df.columns = [col.replace('_mean', '_avg') for col in final_df.columns]

    # 5. Map pose categories
    final_df['Pose'] = final_df['Pose'].map(pose_map)

    # 6. Min-Max normalize numerical columns
    cols_to_norm = final_df.columns.difference(['Sequence_ID', 'Pose'])
    for col in cols_to_norm:
        final_df[col] = min_max_scale(final_df[col])

    # 7. Export
    output_file = f'{action_name}_normalized_summary.csv'
    final_df.to_csv(output_file, index=False)

    print(f"{action_name.replace('_', ' ').title()} processing complete!")
    print(f"Saved: {output_file} ({len(final_df)} sequences summarized)")

    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# MAT DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_mat_metadata(mat_data):
    """
    Extract common metadata from a loaded .mat file.
    Returns (n_frames, action_name, viewpoint).
    """
    n_frames = mat_data['nframes'][0][0]

    action_name = mat_data['action'][0]
    if isinstance(action_name, (list, np.ndarray)):
        action_name = action_name[0]

    viewpoint = mat_data.get('pose', ['Unknown'])[0]
    if isinstance(viewpoint, (list, np.ndarray)):
        viewpoint = viewpoint[0] if len(viewpoint) > 0 else 'Unknown'

    return n_frames, action_name, viewpoint


def load_sequence_as_dataframe(mat_data, n_frames, sequence_id, pose_names=None):
    """
    Load a sequence from mat_data into a DataFrame with NaN for invisible joints.
    Returns a DataFrame with columns: Sequence_ID, Frame, {joint}_x, {joint}_y.
    """
    if pose_names is None:
        pose_names = POSE_NAMES

    raw_data = []
    for f in range(n_frames):
        jx = mat_data['x'][f].astype(float)
        jy = mat_data['y'][f].astype(float)
        vis = mat_data['visibility'][f]

        row = {'Sequence_ID': sequence_id, 'Frame': f + 1}
        for i in range(len(pose_names)):
            if vis[i] == 1:
                row[f'{pose_names[i]}_x'] = jx[i]
                row[f'{pose_names[i]}_y'] = jy[i]
            else:
                row[f'{pose_names[i]}_x'] = np.nan
                row[f'{pose_names[i]}_y'] = np.nan
        raw_data.append(row)

    return pd.DataFrame(raw_data)
