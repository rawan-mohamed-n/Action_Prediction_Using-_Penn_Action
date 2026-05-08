"""
feature_extraction.py — Preprocessed sequences → kinematic feature CSVs.

Runs the full batch pipeline: for each action, iterate all sequences,
preprocess, compute angles, export joints/angles CSVs, then summarize
and normalize into a final feature file.
"""

import os
import numpy as np
import pandas as pd

from config import (
    LABELS_DIR, POSE_NAMES, ANGLE_COLUMNS, DEFAULT_POSE_MAP, ACTIONS,
    joints_csv, angles_csv, summary_csv
)
from helpers import compute_baselines, compute_all_angles, min_max_scale
from preprocessing import load_mat, get_action_name, preprocess_sequence


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def export_joints_and_angles(all_joints_data, all_angles_data, action_name):
    """Export joint and angle DataFrames to CSV files."""
    if not all_joints_data:
        print(f"  No {action_name} sequences found.")
        return

    # Build joint columns
    joint_cols = ['Sequence_ID', 'Frame']
    joint_cols += [f'{n}_x' for n in POSE_NAMES]
    joint_cols += [f'{n}_y' for n in POSE_NAMES]

    joints_df = pd.concat(all_joints_data, ignore_index=True)
    # Keep only the standard columns (drop temp columns like Mid_Hip, etc.)
    available = [c for c in joint_cols if c in joints_df.columns]
    joints_df[available].to_csv(joints_csv(action_name), index=False)

    pd.DataFrame(all_angles_data).reindex(columns=ANGLE_COLUMNS).to_csv(
        angles_csv(action_name), index=False)

    print(f"  Saved: {joints_csv(action_name)}, {angles_csv(action_name)}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZE & NORMALIZE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def summarize_and_normalize(action_name, pose_map=None):
    """
    Full summary pipeline:
      1. Load {action}_joints.csv and {action}_angles.csv
      2. Group angles by (Sequence_ID, Pose) → agg max/min/mean/var
      3. Group joints by Sequence_ID → agg std
      4. Flatten, merge, rename mean→avg, map poses, min-max normalize
      5. Export to {action}_normalized_summary.csv
    """
    if pose_map is None:
        pose_map = DEFAULT_POSE_MAP

    j_file = joints_csv(action_name)
    a_file = angles_csv(action_name)

    if not os.path.exists(j_file) or not os.path.exists(a_file):
        print(f"  Missing CSV files for {action_name}. Run extraction first.")
        return None

    df_joints = pd.read_csv(j_file)
    df_angles = pd.read_csv(a_file)

    # Angle summary: max, min, mean, var per sequence
    angle_cols = [col for col in df_angles.columns if 'Angle' in col]
    angle_summary = df_angles.groupby(['Sequence_ID', 'Pose'])[angle_cols].agg(
        ['max', 'min', 'mean', 'var']
    ).reset_index()
    angle_summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in angle_summary.columns.values
    ]

    # Joint summary: std per sequence
    jcols = [col for col in df_joints.columns if '_x' in col or '_y' in col]
    joint_summary = df_joints.groupby('Sequence_ID')[jcols].agg(['std']).reset_index()
    joint_summary.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in joint_summary.columns.values
    ]

    # Merge and clean
    final_df = pd.merge(angle_summary, joint_summary, on='Sequence_ID', how='inner')
    final_df.columns = [col.replace('_mean', '_avg') for col in final_df.columns]

    # Map pose categories
    final_df['Pose'] = final_df['Pose'].map(pose_map)

    # Min-Max normalize numerical columns
    cols_to_norm = final_df.columns.difference(['Sequence_ID', 'Pose'])
    for col in cols_to_norm:
        final_df[col] = min_max_scale(final_df[col])

    # Export
    out_file = summary_csv(action_name)
    final_df.to_csv(out_file, index=False)
    print(f"  Saved: {out_file} ({len(final_df)} sequences)")

    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# BATCH FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_for_action(action_name):
    """
    Full pipeline for one action:
      1. Iterate all sequences in the configured range
      2. Load .mat, filter by action keyword
      3. Preprocess (symmetry, interpolation)
      4. Compute baselines and angles
      5. Export joints/angles CSVs
      6. Summarize and normalize
    """
    action_cfg = ACTIONS[action_name]
    keywords   = action_cfg['keywords']
    seq_range  = action_cfg['range']

    all_joints_data  = []
    all_angles_data  = []
    processed_count  = 0

    print(f"\n{'='*60}")
    print(f"  Extracting features: {action_name}")
    print(f"{'='*60}")

    for seq_num in range(seq_range[0], seq_range[1]):
        sequence_id = f"{seq_num:04d}"

        # Load .mat
        mat_data = load_mat(LABELS_DIR, sequence_id)
        if mat_data is None:
            continue

        # Check action
        action = get_action_name(mat_data)
        if action not in keywords:
            continue

        # Preprocess
        df, pose_type, base_row = preprocess_sequence(mat_data, sequence_id, action_cfg)

        # Store cleaned joints
        joint_cols = ['Sequence_ID', 'Frame']
        joint_cols += [f'{n}_x' for n in POSE_NAMES]
        joint_cols += [f'{n}_y' for n in POSE_NAMES]
        available = [c for c in joint_cols if c in df.columns]
        all_joints_data.append(df[available].copy().round(3))

        # Compute baselines and angles
        base = compute_baselines(base_row)
        angles = compute_all_angles(df, sequence_id, pose_type, base)
        all_angles_data.extend(angles)

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"  Processed {processed_count} sequences...")

    # Export
    export_joints_and_angles(all_joints_data, all_angles_data, action_name)

    # Summarize and normalize
    summarize_and_normalize(action_name)

    print(f"  Total: {processed_count} {action_name} sequences processed.\n")
    return processed_count


def extract_all():
    """Run feature extraction for all configured actions."""
    for action_name in ACTIONS:
        extract_features_for_action(action_name)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    extract_all()
