"""
preprocessing.py — Raw .mat data → clean, interpolated DataFrames.

Handles loading, symmetry mirroring, and anchor-based interpolation
for all action types.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio

from config import POSE_NAMES, UPPER_JOINTS, LOWER_JOINTS


# ─────────────────────────────────────────────────────────────────────────────
# RAW DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_mat(labels_dir, sequence_id):
    """Load a .mat file and return the mat_data dict, or None if not found."""
    mat_path = os.path.join(labels_dir, f'{sequence_id}.mat')
    if not os.path.exists(mat_path):
        return None
    return sio.loadmat(mat_path)


def get_action_name(mat_data):
    """Extract the action name string from mat_data."""
    action = mat_data['action'][0]
    if isinstance(action, (list, np.ndarray)):
        action = action[0]
    return action


def load_raw_dataframe(mat_data, n_frames, sequence_id):
    """
    Convert raw .mat joint data into a DataFrame.
    Invisible joints (visibility=0) are set to NaN.
    """
    raw_data = []
    for f in range(n_frames):
        jx = mat_data['x'][f].astype(float)
        jy = mat_data['y'][f].astype(float)
        vis = mat_data['visibility'][f]

        row = {'Sequence_ID': sequence_id, 'Frame': f + 1}
        for i in range(len(POSE_NAMES)):
            if vis[i] == 1:
                row[f'{POSE_NAMES[i]}_x'] = jx[i]
                row[f'{POSE_NAMES[i]}_y'] = jy[i]
            else:
                row[f'{POSE_NAMES[i]}_x'] = np.nan
                row[f'{POSE_NAMES[i]}_y'] = np.nan
        raw_data.append(row)

    return pd.DataFrame(raw_data)


# ─────────────────────────────────────────────────────────────────────────────
# MIRROR SYMMETRY (for front-facing poses)
# ─────────────────────────────────────────────────────────────────────────────

def apply_mirror_symmetry(df, n_frames):
    """
    If one side's joint is visible but the other isn't, mirror it across the
    Head's X axis. Works for front-facing poses where L/R are roughly symmetric.
    """
    # Ensure Head is interpolated first
    df['Head_x'] = df['Head_x'].interpolate(method='linear', limit_direction='both')
    df['Head_y'] = df['Head_y'].interpolate(method='linear', limit_direction='both')

    for f in range(n_frames):
        head_x = df.at[f, 'Head_x']

        for joint in ['Sho', 'Elb', 'Wri', 'Hip', 'Kne', 'Ank']:
            r_y, r_x = f'R_{joint}_y', f'R_{joint}_x'
            l_y, l_x = f'L_{joint}_y', f'L_{joint}_x'

            if not pd.isna(df.at[f, r_y]) and pd.isna(df.at[f, l_y]):
                df.at[f, l_y] = df.at[f, r_y]
                df.at[f, l_x] = head_x - (df.at[f, r_x] - head_x)

            elif not pd.isna(df.at[f, l_y]) and pd.isna(df.at[f, r_y]):
                df.at[f, r_y] = df.at[f, l_y]
                df.at[f, r_x] = head_x + (head_x - df.at[f, l_x])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATION STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def anchor_interpolate_single(df):
    """
    Single-anchor interpolation: all joints relative to the Head.
    Used for jump-rope and pullup.
    """
    df['Head_x'] = df['Head_x'].interpolate(limit_direction='both')
    df['Head_y'] = df['Head_y'].interpolate(limit_direction='both')

    for name in POSE_NAMES[1:]:  # Skip Head itself
        df[f'{name}_rel_x'] = df[f'{name}_x'] - df['Head_x']
        df[f'{name}_rel_y'] = df[f'{name}_y'] - df['Head_y']

        df[f'{name}_rel_x'] = df[f'{name}_rel_x'].interpolate(limit_direction='both')
        df[f'{name}_rel_y'] = df[f'{name}_rel_y'].interpolate(limit_direction='both')

        df[f'{name}_x'] = df[f'{name}_rel_x'] + df['Head_x']
        df[f'{name}_y'] = df[f'{name}_rel_y'] + df['Head_y']

        df.drop(columns=[f'{name}_rel_x', f'{name}_rel_y'], inplace=True, errors='ignore')

    return df


def anchor_interpolate_dual(df):
    """
    Dual-anchor interpolation: upper body relative to Head,
    lower body relative to Mid-Hip. Used for squat and pushup.
    """
    df['Head_x'] = df['Head_x'].interpolate(method='linear', limit_direction='both')
    df['Head_y'] = df['Head_y'].interpolate(method='linear', limit_direction='both')

    # Establish Mid-Hip anchor
    df['Mid_Hip_x'] = df[['L_Hip_x', 'R_Hip_x']].mean(axis=1).interpolate(
        method='linear', limit_direction='both')
    df['Mid_Hip_y'] = df[['L_Hip_y', 'R_Hip_y']].mean(axis=1).interpolate(
        method='linear', limit_direction='both')

    # Upper body → Head anchor
    for name in UPPER_JOINTS:
        df[f'{name}_rel_x'] = df[f'{name}_x'] - df['Head_x']
        df[f'{name}_rel_y'] = df[f'{name}_y'] - df['Head_y']
        df[f'{name}_rel_x'] = df[f'{name}_rel_x'].interpolate(
            method='linear', limit_direction='both')
        df[f'{name}_rel_y'] = df[f'{name}_rel_y'].interpolate(
            method='linear', limit_direction='both')
        df[f'{name}_x'] = df[f'{name}_rel_x'] + df['Head_x']
        df[f'{name}_y'] = df[f'{name}_rel_y'] + df['Head_y']

    # Lower body → Mid-Hip anchor
    for name in LOWER_JOINTS:
        df[f'{name}_rel_x'] = df[f'{name}_x'] - df['Mid_Hip_x']
        df[f'{name}_rel_y'] = df[f'{name}_y'] - df['Mid_Hip_y']
        df[f'{name}_rel_x'] = df[f'{name}_rel_x'].interpolate(
            method='linear', limit_direction='both')
        df[f'{name}_rel_y'] = df[f'{name}_rel_y'].interpolate(
            method='linear', limit_direction='both')
        df[f'{name}_x'] = df[f'{name}_rel_x'] + df['Mid_Hip_x']
        df[f'{name}_y'] = df[f'{name}_rel_y'] + df['Mid_Hip_y']

    return df


# ─────────────────────────────────────────────────────────────────────────────
# VIEWPOINT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_viewpoint(df, threshold=0.45):
    """
    Detect whether a sequence is Front/Back or Side view based on
    shoulder width vs torso height ratio.
    """
    avg_sw = abs(df['L_Sho_x'] - df['R_Sho_x']).mean()

    # Use best available torso height estimate
    if 'Mid_Hip_y' in df.columns:
        avg_th = abs(
            df[['L_Sho_y', 'R_Sho_y']].mean(axis=1) -
            df[['L_Hip_y', 'R_Hip_y']].mean(axis=1)
        ).mean()
    else:
        avg_th = abs(df['Head_y'] - df['L_Hip_y']).mean()

    return 'Front/Back' if avg_sw > (avg_th * threshold) else 'Side'


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE FRAME SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_baseline_frame_standing(df):
    """
    Find the "standing" frame — maximum total height.
    Used for squat, pushup, jump-rope.
    """
    if 'Mid_Hip_y' not in df.columns:
        df['Mid_Hip_y'] = df[['L_Hip_y', 'R_Hip_y']].mean(axis=1)

    df['Mid_Ank_y'] = df[['L_Ank_y', 'R_Ank_y']].mean(axis=1)
    df['Total_Height'] = (abs(df['Head_y'] - df['Mid_Hip_y']) +
                          abs(df['Mid_Hip_y'] - df['Mid_Ank_y']))

    if df['Total_Height'].isna().all():
        df['Total_Height'] = abs(df['Head_y'] - df['Mid_Hip_y'])

    return df['Total_Height'].idxmax() if not df['Total_Height'].isna().all() else 0


def find_baseline_frame_hanging(df):
    """
    Find the "hanging" frame — maximum distance from Head to Wrists.
    Used for pullup.
    """
    df['Mid_Wri_y'] = df[['L_Wri_y', 'R_Wri_y']].mean(axis=1)
    df['Hang_Dist'] = abs(df['Mid_Wri_y'] - df['Head_y'])

    return df['Hang_Dist'].idxmax() if not df['Hang_Dist'].isna().all() else 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREPROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_sequence(mat_data, sequence_id, action_config):
    """
    Full preprocessing pipeline for a single sequence.

    Parameters:
        mat_data:       loaded .mat dict
        sequence_id:    string ID
        action_config:  dict from config.ACTIONS[action_name]

    Returns:
        df:         cleaned DataFrame with interpolated joints
        pose_type:  'Front/Back' or 'Side'
        base_row:   the baseline reference row (for angle computation)
    """
    n_frames = mat_data['nframes'][0][0]

    # 1. Load raw data
    df = load_raw_dataframe(mat_data, n_frames, sequence_id)

    # 2. Mirror symmetry (if enabled)
    if action_config.get('use_symmetry', True):
        df = apply_mirror_symmetry(df, n_frames)

    # 3. Anchor-based interpolation
    interp_method = action_config.get('interpolation', 'single')
    if interp_method == 'dual':
        df = anchor_interpolate_dual(df)
    else:
        df = anchor_interpolate_single(df)

    # 4. Viewpoint detection
    threshold = action_config.get('viewpoint_threshold', 0.45)
    pose_type = detect_viewpoint(df, threshold)

    # 5. Baseline frame
    baseline_method = action_config.get('baseline_method', 'standing')
    if baseline_method == 'hanging':
        stand_idx = find_baseline_frame_hanging(df)
    else:
        stand_idx = find_baseline_frame_standing(df)

    base_row = df.iloc[stand_idx]

    return df, pose_type, base_row
