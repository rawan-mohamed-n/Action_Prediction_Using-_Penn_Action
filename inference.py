"""
inference.py — Real-time action classification & form quality scoring.

Uses a threaded webcam capture, MediaPipe Pose for landmark detection,
and the models trained by form_evaluator_training.py:
  - scaler.pkl           → StandardScaler
  - action_classifier.pkl → RandomForest action classifier
  - quality_regressor.pkl → RandomForest quality regressor

All feature-schema constants are imported from config.py.
"""

import argparse
import logging
import queue
import threading
import time
from collections import deque

import cv2
import joblib
import numpy as np
import pandas as pd

from config import (
    ANGLE_KEYS,
    LANDMARK_KEYS,
    FEATURE_COLS,
    ACTION_OHE_COLS,
    IDLE_Y_KEYS,
    IDLE_VARIANCE_THRESHOLD,
    IDLE_WINDOW_SIZE,
    INFERENCE_EVERY_N_FRAMES,
    ACTION_WINDOW_SIZE,
    MAIN_BUFFER_MAXLEN,
    SCALER_PATH,
    ACTION_MODEL_PATH,
    QUALITY_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# MediaPipe compatibility shim
# ---------------------------------------------------------------------------
import mediapipe as mp

try:
    mp_solutions = mp.solutions
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    try:
        from mediapipe.python import solutions as mp_solutions
        from mediapipe.python.solutions import drawing_utils as mp_drawing
    except Exception as exc:
        raise ImportError(
            "mediapipe does not expose `solutions`. "
            "Reinstall: python -m pip install --upgrade mediapipe"
        ) from exc

POSE_LANDMARK = mp_solutions.pose.PoseLandmark

# ---------------------------------------------------------------------------
# OpenCV backend map
# ---------------------------------------------------------------------------
BACKEND_MAP: dict = {
    'any':    None,
    'msmf':   getattr(cv2, 'CAP_MSMF',  None),
    'dshow':  getattr(cv2, 'CAP_DSHOW', None),
    'ffmpeg': getattr(cv2, 'CAP_FFMPEG', None),
}


# ===========================================================================
# 1. Camera Threading
# ===========================================================================

class VideoCaptureThread:
    """
    Daemon thread that continuously reads frames from a video source and
    places the latest frame in a single-slot queue.  The main thread never
    blocks on I/O — it always gets the most recent frame or None.
    """

    def __init__(
        self,
        src,
        width: int | None = None,
        height: int | None = None,
        queue_size: int = 1,
        backend=None,
    ) -> None:
        self.src = src
        self.width = width
        self.height = height
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._capture: cv2.VideoCapture | None = None
        self._backend = backend

    def start(self) -> 'VideoCaptureThread':
        self._capture = (
            cv2.VideoCapture(self.src)
            if self._backend is None
            else cv2.VideoCapture(self.src, self._backend)
        )
        if not self._capture.isOpened():
            logging.error('Failed to open video source: %s', self.src)
        if self.width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self._thread.start()
        return self

    def _reader(self) -> None:
        while not self._stop_event.is_set():
            if self._capture is None:
                time.sleep(0.005)
                continue
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

    def read(self, timeout: float = 0.5):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture is not None:
            self._capture.release()


# ===========================================================================
# Geometry helpers
# ===========================================================================

def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Three-point joint angle (degrees) at vertex b.  Returns NaN on bad input."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0.0:
        return np.nan
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return np.array([np.nan, np.nan])
    return (a + b) * 0.5


def _safe_stat(values: np.ndarray, func) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(func(v)) if v.size > 0 else np.nan


# ===========================================================================
# Feature extraction (per-frame row)
# ===========================================================================

def extract_pose_features(
    landmarks,
    frame_width: int,
    frame_height: int,
    visibility_thresh: float = 0.5,
) -> dict[str, float]:
    """
    Extract one raw feature row from a MediaPipe landmark list.
    Returns angles (8) + xy-coords (14 × 2) = 36 values per frame.
    """

    def pt(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        if lm.visibility < visibility_thresh:
            return np.array([np.nan, np.nan])
        return np.array([lm.x * frame_width, lm.y * frame_height], dtype=float)

    head    = pt(POSE_LANDMARK.NOSE.value)
    l_sho   = pt(POSE_LANDMARK.LEFT_SHOULDER.value)
    r_sho   = pt(POSE_LANDMARK.RIGHT_SHOULDER.value)
    l_elb   = pt(POSE_LANDMARK.LEFT_ELBOW.value)
    r_elb   = pt(POSE_LANDMARK.RIGHT_ELBOW.value)
    l_wri   = pt(POSE_LANDMARK.LEFT_WRIST.value)
    r_wri   = pt(POSE_LANDMARK.RIGHT_WRIST.value)
    l_hip   = pt(POSE_LANDMARK.LEFT_HIP.value)
    r_hip   = pt(POSE_LANDMARK.RIGHT_HIP.value)
    l_kne   = pt(POSE_LANDMARK.LEFT_KNEE.value)
    r_kne   = pt(POSE_LANDMARK.RIGHT_KNEE.value)
    l_ank   = pt(POSE_LANDMARK.LEFT_ANKLE.value)
    r_ank   = pt(POSE_LANDMARK.RIGHT_ANKLE.value)
    mid_hip = _midpoint(l_hip, r_hip)

    row: dict[str, float] = {
        'L_Shoulder_Angle': _calc_angle(l_elb, l_sho, l_hip),
        'R_Shoulder_Angle': _calc_angle(r_elb, r_sho, r_hip),
        'L_Elbow_Angle':    _calc_angle(l_sho, l_elb, l_wri),
        'R_Elbow_Angle':    _calc_angle(r_sho, r_elb, r_wri),
        'L_Hip_Angle':      _calc_angle(l_sho, l_hip, l_kne),
        'R_Hip_Angle':      _calc_angle(r_sho, r_hip, r_kne),
        'L_Knee_Angle':     _calc_angle(l_hip, l_kne, l_ank),
        'R_Knee_Angle':     _calc_angle(r_hip, r_kne, r_ank),
    }

    for key, point in zip(
        LANDMARK_KEYS,
        [head, l_sho, r_sho, l_elb, r_elb, l_wri, r_wri,
         l_hip, r_hip, l_kne, r_kne, l_ank, r_ank, mid_hip],
    ):
        row[f'{key}_x'] = float(point[0])
        row[f'{key}_y'] = float(point[1])

    return row


def empty_feature_row() -> dict[str, float]:
    """Sentinel row used when no subject is detected."""
    row = {k: 0.0 for k in ANGLE_KEYS}
    for key in LANDMARK_KEYS:
        row[f'{key}_x'] = 0.0
        row[f'{key}_y'] = 0.0
    return row


# ===========================================================================
# Idle State detector
# ===========================================================================

def compute_idle_variance(recent_rows: list[dict]) -> float:
    """
    Sum of per-key variance of Y-coordinates across the provided rows.
    A value < IDLE_VARIANCE_THRESHOLD indicates the subject is stationary.
    """
    df = pd.DataFrame(recent_rows)
    return float(
        np.nansum([
            np.nanvar(df[k].values)
            for k in IDLE_Y_KEYS
            if k in df.columns
        ])
    )


# ===========================================================================
# Feature aggregation
# ===========================================================================

def build_window_features(buffer_rows: list[dict]) -> pd.DataFrame:
    """
    Aggregate a list of raw per-frame rows into the single summary-stat row
    expected by both models.  Column order strictly matches FEATURE_COLS.
    """
    df_buf = pd.DataFrame(buffer_rows)
    feature_dict: dict[str, float] = {}

    for angle_key in ANGLE_KEYS:
        vals = df_buf[angle_key].values
        feature_dict[f'{angle_key}_max'] = _safe_stat(vals, np.max)
        feature_dict[f'{angle_key}_min'] = _safe_stat(vals, np.min)
        feature_dict[f'{angle_key}_avg'] = _safe_stat(vals, np.mean)
        feature_dict[f'{angle_key}_var'] = _safe_stat(vals, np.var)

    for lm_key in LANDMARK_KEYS:
        feature_dict[f'{lm_key}_x_std'] = _safe_stat(
            df_buf[f'{lm_key}_x'].values, np.std
        )
        feature_dict[f'{lm_key}_y_std'] = _safe_stat(
            df_buf[f'{lm_key}_y'].values, np.std
        )

    return pd.DataFrame([feature_dict], columns=FEATURE_COLS).fillna(0.0)


def build_quality_input(
    base_features: pd.DataFrame,
    predicted_action: str,
) -> pd.DataFrame:
    """
    Append one-hot action encoding to base features.
    Output shape: (1, 64) — matches quality_regressor training input.
    """
    ohe = pd.DataFrame(
        [[0] * len(ACTION_OHE_COLS)],
        columns=ACTION_OHE_COLS,
    )
    action_col = f'action_{predicted_action}'
    if action_col in ohe.columns:
        ohe.loc[0, action_col] = 1
    return pd.concat([base_features.reset_index(drop=True), ohe], axis=1)


# ===========================================================================
# Utility
# ===========================================================================

def _parse_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def _resolve_backend(name: str, source):
    if not name:
        return None
    name = name.lower()
    if name == 'auto':
        name = 'ffmpeg' if (isinstance(source, str) and '://' in source) else 'dshow'
    backend = BACKEND_MAP.get(name)
    if backend is None and name not in ('any', 'dshow'):
        logging.warning(
            "Requested backend '%s' unavailable — falling back to default.", name
        )
    return backend


def _draw_overlay(
    frame: np.ndarray,
    status: str,
    action: str,
    quality: float,
    fps: float,
) -> None:
    """Render a clean HUD on the frame in-place."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (370, 115), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if status in {'Status: Idle', 'Status: No Subject'}:
        quality_str = 'N/A'
    else:
        quality_str = f'{quality:.2f}' if not np.isnan(quality) else '--'

    cv2.putText(frame, status,                (18, 33),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 220), 2)
    cv2.putText(frame, f'Action : {action}',  (18, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.68, (50, 255, 80),  2)
    cv2.putText(frame, f'Quality: {quality_str}', (18, 87),  cv2.FONT_HERSHEY_SIMPLEX, 0.68, (50, 255, 80),  2)
    cv2.putText(frame, f'FPS: {fps:5.1f}',   (18, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(
        description='Real-time Action & Form Inference'
    )
    parser.add_argument('--source',  default='1',
                        help='Webcam index or DroidCam/RTSP URL')
    parser.add_argument('--width',   type=int,   default=None)
    parser.add_argument('--height',  type=int,   default=None)
    parser.add_argument('--buffer',  type=int,   default=MAIN_BUFFER_MAXLEN,
                        help='Main deque length (default: 90 frames)')
    parser.add_argument('--backend', default='auto',
                        choices=['auto', 'any', 'msmf', 'dshow', 'ffmpeg'])
    args = parser.parse_args()

    # ── Load models ─────────────────────────────────────────────────────────
    for path, label in [
        (SCALER_PATH,        'scaler'),
        (ACTION_MODEL_PATH,  'action_classifier'),
        (QUALITY_MODEL_PATH, 'quality_regressor'),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f'{label} not found at {path}. '
                'Run: python run_pipeline.py --form-train'
            )

    scaler        = joblib.load(SCALER_PATH)
    action_model  = joblib.load(ACTION_MODEL_PATH)
    quality_model = joblib.load(QUALITY_MODEL_PATH)

    # Validate feature counts
    expected_action  = len(FEATURE_COLS)
    expected_quality = len(FEATURE_COLS) + len(ACTION_OHE_COLS)
    actual_action    = getattr(action_model,  'n_features_in_', expected_action)
    actual_quality   = getattr(quality_model, 'n_features_in_', expected_quality)

    if actual_action != expected_action:
        logging.warning(
            'Action model expects %d features; FEATURE_COLS has %d.',
            actual_action, expected_action,
        )
    if actual_quality != expected_quality:
        logging.warning(
            'Quality model expects %d features; script produces %d.',
            actual_quality, expected_quality,
        )

    logging.info(
        'Models loaded.  Action features: %d | Quality features: %d',
        expected_action, expected_quality,
    )

    # ── Camera ──────────────────────────────────────────────────────────────
    source  = _parse_source(args.source)
    backend = _resolve_backend(args.backend, source)
    capture = VideoCaptureThread(
        source, width=args.width, height=args.height, backend=backend,
    ).start()

    # ── Buffers & state ─────────────────────────────────────────────────────
    buffer: deque[dict]  = deque(maxlen=args.buffer)
    buffer_lock          = threading.Lock()
    inference_stop       = threading.Event()
    inference_ready      = False
    buffer_counter       = 0

    mp_pose = mp_solutions.pose
    mp_draw = mp_drawing
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    predicted_action:  str   = '--'
    predicted_quality: float = np.nan
    status:            str   = 'Status: Warming'
    frame_counter:     int   = 0

    # ── Inference worker ────────────────────────────────────────────────────
    def inference_worker() -> None:
        nonlocal predicted_action, predicted_quality, inference_ready, buffer_counter
        last_counter = 0
        while not inference_stop.is_set():
            with buffer_lock:
                ready = inference_ready
                counter = buffer_counter
                buffer_len = len(buffer)
                if (
                    ready
                    and buffer_len >= buffer.maxlen
                    and (counter - last_counter) >= INFERENCE_EVERY_N_FRAMES
                ):
                    action_rows  = list(buffer)[-ACTION_WINDOW_SIZE:]
                    quality_rows = list(buffer)
                else:
                    action_rows = None

            if action_rows is None:
                time.sleep(0.005)
                continue

            action_features = build_window_features(action_rows)
            action_scaled = pd.DataFrame(
                scaler.transform(action_features), columns=FEATURE_COLS,
            )
            action_pred = action_model.predict(action_scaled)[0]

            quality_features = build_window_features(quality_rows)
            quality_scaled = pd.DataFrame(
                scaler.transform(quality_features), columns=FEATURE_COLS,
            )
            quality_input = build_quality_input(quality_scaled, action_pred)
            quality_pred = float(quality_model.predict(quality_input)[0])

            with buffer_lock:
                predicted_action  = action_pred
                predicted_quality = quality_pred
                last_counter      = counter

    # FPS tracking
    fps_start   = time.perf_counter()
    fps_count   = 0
    current_fps = 0.0

    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    logging.info("Starting inference loop. Press 'q' to quit.")

    try:
        while True:
            frame = capture.read()
            if frame is None:
                continue

            frame_counter += 1
            fps_count     += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count   = 0
                fps_start   = time.perf_counter()

            # ── MediaPipe Pose ───────────────────────────────────────────
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # ── No subject ───────────────────────────────────────────────
            if not results.pose_landmarks:
                status = 'Status: No Subject'
                with buffer_lock:
                    buffer.append(empty_feature_row())
                    buffer_counter += 1
                    inference_ready = False
                    action  = predicted_action
                    quality = predicted_quality
                _draw_overlay(frame, status, action, quality, current_fps)
                cv2.imshow('Action & Form Evaluation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Subject detected — draw skeleton
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(50, 255, 80),   thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=2),
            )

            row = extract_pose_features(
                results.pose_landmarks.landmark,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )

            with buffer_lock:
                buffer.append(row)
                buffer_counter += 1
                buffer_len = len(buffer)
                recent = (
                    list(buffer)[-IDLE_WINDOW_SIZE:]
                    if buffer_len >= IDLE_WINDOW_SIZE
                    else None
                )

            # ── Idle check ───────────────────────────────────────────────
            if recent is not None:
                if compute_idle_variance(recent) < IDLE_VARIANCE_THRESHOLD:
                    status = 'Status: Idle'
                    with buffer_lock:
                        inference_ready = False
                        action  = predicted_action
                        quality = predicted_quality
                    _draw_overlay(frame, status, action, quality, current_fps)
                    cv2.imshow('Action & Form Evaluation', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

            # ── Warming ──────────────────────────────────────────────────
            if buffer_len < buffer.maxlen:
                status = 'Status: Warming'
                with buffer_lock:
                    inference_ready = False
                    action  = predicted_action
                    quality = predicted_quality
                _draw_overlay(frame, status, action, quality, current_fps)
                cv2.imshow('Action & Form Evaluation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ── Running ──────────────────────────────────────────────────
            status = 'Status: Running'
            with buffer_lock:
                inference_ready = True
                action  = predicted_action
                quality = predicted_quality

            _draw_overlay(frame, status, action, quality, current_fps)
            cv2.imshow('Action & Form Evaluation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        inference_stop.set()
        if worker is not None:
            worker.join(timeout=1.0)
        capture.stop()
        pose.close()
        cv2.destroyAllWindows()
        logging.info('Shutdown complete.')


if __name__ == '__main__':
    main()
