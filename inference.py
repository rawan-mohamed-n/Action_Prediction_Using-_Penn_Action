import argparse
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Model & scaler — loaded at module level so the worker thread never touches
# the filesystem after startup.
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent

_scaler_path = MODEL_DIR / "scaler.pkl"
_action_path = MODEL_DIR / "action_classifier.pkl"
_quality_path = MODEL_DIR / "quality_regressor.pkl"

for _p in (_scaler_path, _action_path, _quality_path):
    if not _p.exists():
        raise FileNotFoundError(
            f"Required artefact not found: {_p}\n"
            "Run form_evaluator.ipynb first to generate all .pkl files."
        )

scaler        = joblib.load(_scaler_path)
action_model  = joblib.load(_action_path)
quality_model = joblib.load(_quality_path)

# ---------------------------------------------------------------------------
# MediaPipe compatibility shim
# ---------------------------------------------------------------------------
try:
    mp_solutions = mp.solutions
    mp_drawing   = mp.solutions.drawing_utils
except AttributeError:
    try:
        from mediapipe.python import solutions as mp_solutions
        from mediapipe.python.solutions import drawing_utils as mp_drawing
    except Exception as _exc:
        raise ImportError(
            "mediapipe does not expose `solutions`. "
            "Reinstall: python -m pip install --upgrade mediapipe"
        ) from _exc

POSE_LANDMARK = mp_solutions.pose.PoseLandmark

# ---------------------------------------------------------------------------
# Feature schema  —  must exactly match form_evaluator.ipynb column order
# 60 base columns: 32 angle stats  +  28 landmark-coord stds
# ---------------------------------------------------------------------------
ANGLE_KEYS: list[str] = [
    "L_Shoulder_Angle", "R_Shoulder_Angle",
    "L_Elbow_Angle",    "R_Elbow_Angle",
    "L_Hip_Angle",      "R_Hip_Angle",
    "L_Knee_Angle",     "R_Knee_Angle",
]

LANDMARK_KEYS: list[str] = [
    "Head",  "L_Sho", "R_Sho",
    "L_Elb", "R_Elb", "L_Wri", "R_Wri",
    "L_Hip", "R_Hip", "L_Kne", "R_Kne",
    "L_Ank", "R_Ank", "Mid_Hip",
]

FEATURE_COLS: list[str] = [
    # 8 angles × 4 stats = 32
    "L_Shoulder_Angle_max", "L_Shoulder_Angle_min",
    "L_Shoulder_Angle_avg", "L_Shoulder_Angle_var",
    "R_Shoulder_Angle_max", "R_Shoulder_Angle_min",
    "R_Shoulder_Angle_avg", "R_Shoulder_Angle_var",
    "L_Elbow_Angle_max",    "L_Elbow_Angle_min",
    "L_Elbow_Angle_avg",    "L_Elbow_Angle_var",
    "R_Elbow_Angle_max",    "R_Elbow_Angle_min",
    "R_Elbow_Angle_avg",    "R_Elbow_Angle_var",
    "L_Hip_Angle_max",      "L_Hip_Angle_min",
    "L_Hip_Angle_avg",      "L_Hip_Angle_var",
    "R_Hip_Angle_max",      "R_Hip_Angle_min",
    "R_Hip_Angle_avg",      "R_Hip_Angle_var",
    "L_Knee_Angle_max",     "L_Knee_Angle_min",
    "L_Knee_Angle_avg",     "L_Knee_Angle_var",
    "R_Knee_Angle_max",     "R_Knee_Angle_min",
    "R_Knee_Angle_avg",     "R_Knee_Angle_var",
    # 14 landmarks × 2 axes = 28 (interleaved x/y, Mid_Hip last)
    "Head_x_std",    "Head_y_std",
    "L_Sho_x_std",   "L_Sho_y_std",
    "R_Sho_x_std",   "R_Sho_y_std",
    "L_Elb_x_std",   "L_Elb_y_std",
    "R_Elb_x_std",   "R_Elb_y_std",
    "L_Wri_x_std",   "L_Wri_y_std",
    "R_Wri_x_std",   "R_Wri_y_std",
    "L_Hip_x_std",   "L_Hip_y_std",
    "R_Hip_x_std",   "R_Hip_y_std",
    "L_Kne_x_std",   "L_Kne_y_std",
    "R_Kne_x_std",   "R_Kne_y_std",
    "L_Ank_x_std",   "L_Ank_y_std",
    "R_Ank_x_std",   "R_Ank_y_std",
    "Mid_Hip_x_std", "Mid_Hip_y_std",
]

# OHE columns — pd.get_dummies sorts alphabetically; order must match training
ACTIONS: list[str]        = ["jump_rope", "pullup", "pushup", "squat"]
ACTION_OHE_COLS: list[str] = [f"action_{a}" for a in sorted(ACTIONS)]

assert len(FEATURE_COLS) == 60, "FEATURE_COLS must be exactly 60 columns"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# ── Idle detection ───────────────────────────────────────────────────────────
# Window size: 20 frames ≈ 667 ms at 30 fps.
# Large enough to span a partial rep so short stillness at movement extremes
# (squat bottom, pullup top, jump peak) doesn't dominate the measurement.
IDLE_WINDOW_SIZE: int = 20

# Coordinate keys checked for motion.
# Using peak-to-peak RANGE (not variance) across the window:
#   variance  → low during slow steady descent even when person is moving
#   range     → captures total displacement; any rep shows 50 + px of travel
#
# Keys cover all four exercises:
#   Y-axis: hips (squat/jump), shoulders (pullup/pushup), wrists (jump rope),
#           knees (squat depth)
#   X-axis: wrists (jump rope lateral arc), hips (squat lateral sway)
IDLE_MOTION_KEYS: list[str] = [
    "L_Hip_y",  "R_Hip_y",    # squat depth, jump height
    "L_Sho_y",  "R_Sho_y",    # pullup travel, pushup depth
    "L_Wri_y",  "R_Wri_y",    # jump rope wrist rotation
    "L_Kne_y",  "R_Kne_y",    # squat knee bend
    "L_Wri_x",  "R_Wri_x",    # jump rope lateral swing
    "L_Hip_x",  "R_Hip_x",    # lateral sway / weight shift
]

# Maximum single-key range (px) below which the subject is considered idle.
# PIXEL-SPACE (lm.x * frame_width — default in this script):
#   640 × 480  → 30 px  (standing jitter ≈ 10–20 px; any exercise ≥ 40 px)
#   1280 × 720 → 60 px
# NORMALISED [0,1] coords (remove frame_width/height multiply):
#   use 0.04–0.08
IDLE_RANGE_THRESHOLD_DEFAULT: float = 30.0

# ── Continuous quality EMA ────────────────────────────────────────────────────
# The worker fires every INFERENCE_EVERY_N_FRAMES and writes a raw prediction.
# The main thread blends toward that target every rendered frame:
#     ema = α × raw + (1 − α) × ema
# Higher α → faster response, more jitter.
# Lower  α → smoother, more lag.
# 0.15 converges ~55% of the way to a new value within 5 frames (one inference
# cycle), giving smooth display without perceivable delay.
EMA_ALPHA_DEFAULT: float = 0.15

# ML pipeline runs every Nth frame; in between, last predictions are displayed
INFERENCE_EVERY_N_FRAMES: int = 5

# Stage 1 (action): most-recent 30-frame slice of the main buffer
ACTION_WINDOW_SIZE: int = 30

# Stage 2 (quality): full deque depth
MAIN_BUFFER_MAXLEN: int = 90

# OpenCV capture backend map
BACKEND_MAP: dict = {
    "any":    None,
    "msmf":   getattr(cv2, "CAP_MSMF",  None),
    "dshow":  getattr(cv2, "CAP_DSHOW", None),
    "ffmpeg": getattr(cv2, "CAP_FFMPEG", None),
}

# ---------------------------------------------------------------------------
# Validate loaded models against the expected feature contract at startup
# ---------------------------------------------------------------------------
_n_action  = getattr(action_model,  "n_features_in_", len(FEATURE_COLS))
_n_quality = getattr(quality_model, "n_features_in_", len(FEATURE_COLS) + len(ACTION_OHE_COLS))

if _n_action != len(FEATURE_COLS):
    logging.warning(
        "action_model expects %d features; script produces %d — check FEATURE_COLS order.",
        _n_action, len(FEATURE_COLS),
    )
if _n_quality != len(FEATURE_COLS) + len(ACTION_OHE_COLS):
    logging.warning(
        "quality_model expects %d features; script produces %d — check column order.",
        _n_quality, len(FEATURE_COLS) + len(ACTION_OHE_COLS),
    )


# ===========================================================================
# Camera Thread
# ===========================================================================
class VideoCaptureThread:
    """
    Daemon thread — continuously drains the camera into a 1-slot queue.
    The main thread always gets the freshest frame without blocking on I/O.
    """

    def __init__(
        self,
        src,
        width:      int | None = None,
        height:     int | None = None,
        queue_size: int        = 1,
        backend                = None,
    ) -> None:
        self.src     = src
        self.width   = width
        self.height  = height
        self._queue       : queue.Queue           = queue.Queue(maxsize=queue_size)
        self._stop_event  : threading.Event       = threading.Event()
        self._thread      : threading.Thread      = threading.Thread(
            target=self._reader, daemon=True, name="CameraThread"
        )
        self._capture     : cv2.VideoCapture | None = None
        self._backend     = backend

    def start(self) -> "VideoCaptureThread":
        self._capture = (
            cv2.VideoCapture(self.src)
            if self._backend is None
            else cv2.VideoCapture(self.src, self._backend)
        )
        if not self._capture.isOpened():
            logging.error("Failed to open video source: %s", self.src)
        if self.width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH,  float(self.width))
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
            # Evict stale frame so the main thread always gets the freshest
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
    """Three-point joint angle in degrees at vertex b.  NaN on degenerate input."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    ba, bc = a - b, c - b
    denom  = np.linalg.norm(ba) * np.linalg.norm(bc)
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
# Per-frame feature extraction
# ===========================================================================

def extract_pose_features(
    landmarks,
    frame_width:      int,
    frame_height:     int,
    visibility_thresh: float = 0.5,
) -> dict[str, float]:
    """
    Convert one MediaPipe landmark set into a raw feature row (36 values).
    Coordinates are stored in PIXEL space so the trained scaler maps them
    into the same z-score distribution as the raw joint CSV training data.
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
        "L_Shoulder_Angle": _calc_angle(l_elb, l_sho, l_hip),
        "R_Shoulder_Angle": _calc_angle(r_elb, r_sho, r_hip),
        "L_Elbow_Angle":    _calc_angle(l_sho, l_elb, l_wri),
        "R_Elbow_Angle":    _calc_angle(r_sho, r_elb, r_wri),
        "L_Hip_Angle":      _calc_angle(l_sho, l_hip, l_kne),
        "R_Hip_Angle":      _calc_angle(r_sho, r_hip, r_kne),
        "L_Knee_Angle":     _calc_angle(l_hip, l_kne, l_ank),
        "R_Knee_Angle":     _calc_angle(r_hip, r_kne, r_ank),
    }

    for key, point in zip(
        LANDMARK_KEYS,
        [head,  l_sho, r_sho, l_elb, r_elb, l_wri, r_wri,
         l_hip, r_hip, l_kne, r_kne, l_ank, r_ank, mid_hip],
    ):
        row[f"{key}_x"] = float(point[0])
        row[f"{key}_y"] = float(point[1])

    return row


def empty_feature_row() -> dict[str, float]:
    """Zero-filled sentinel used when no subject is visible (keeps timeline intact)."""
    row: dict[str, float] = {k: 0.0 for k in ANGLE_KEYS}
    for key in LANDMARK_KEYS:
        row[f"{key}_x"] = 0.0
        row[f"{key}_y"] = 0.0
    return row


# ===========================================================================
# Heuristic Gatekeeper — Idle State detector
# ===========================================================================

# (IDLE_MOTION_KEYS defined in the constants block above)

def compute_motion_range(recent_rows: list[dict]) -> float:
    """
    Returns the MAXIMUM peak-to-peak range (max − min) across all tracked
    coordinate keys over the provided window.

    A value below IDLE_RANGE_THRESHOLD → subject is stationary.

    Why range instead of variance:
      Variance is low during slow, steady movement (each consecutive frame
      barely differs from the last).  Range captures the *total displacement*
      across the window regardless of how smooth the movement is — a squat
      descent shows 80–150 px of hip travel over 20 frames even if each frame
      only moves 5 px.

    Why max-across-keys instead of sum:
      Sum would let many small movements add up to cross the threshold.
      Max is a clean logical OR: if ANY joint moved significantly, the person
      is not idle — no matter what the other joints did.
    """
    df = pd.DataFrame(recent_rows)
    max_range: float = 0.0
    for key in IDLE_MOTION_KEYS:
        if key not in df.columns:
            continue
        vals = df[key].dropna().values
        if len(vals) < 2:
            continue
        key_range = float(vals.max() - vals.min())
        if key_range > max_range:
            max_range = key_range
    return max_range


# ===========================================================================
# Feature aggregation (called exclusively on the background worker thread)
# ===========================================================================

def build_window_features(buffer_rows: list[dict]) -> pd.DataFrame:
    """
    Aggregate N raw per-frame rows into the single (1 × 60) summary row
    expected by both models.  Column order strictly matches FEATURE_COLS.
    Residual NaNs are zeroed — mirrors the training imputer's fill_value=0.
    """
    df_buf = pd.DataFrame(buffer_rows)
    feature_dict: dict[str, float] = {}

    for angle_key in ANGLE_KEYS:
        vals = df_buf[angle_key].values
        feature_dict[f"{angle_key}_max"] = _safe_stat(vals, np.max)
        feature_dict[f"{angle_key}_min"] = _safe_stat(vals, np.min)
        feature_dict[f"{angle_key}_avg"] = _safe_stat(vals, np.mean)
        feature_dict[f"{angle_key}_var"] = _safe_stat(vals, np.var)

    for lm_key in LANDMARK_KEYS:
        feature_dict[f"{lm_key}_x_std"] = _safe_stat(df_buf[f"{lm_key}_x"].values, np.std)
        feature_dict[f"{lm_key}_y_std"] = _safe_stat(df_buf[f"{lm_key}_y"].values, np.std)

    return pd.DataFrame([feature_dict], columns=FEATURE_COLS).fillna(0.0)


def build_quality_input(
    scaled_features: pd.DataFrame,
    predicted_action: str,
) -> pd.DataFrame:
    """
    Append one-hot action encoding to the already-scaled (1 × 60) feature row.
    Output: (1 × 64) — matches quality_regressor training input exactly.
    """
    ohe = pd.DataFrame([[0] * len(ACTION_OHE_COLS)], columns=ACTION_OHE_COLS)
    action_col = f"action_{predicted_action}"
    if action_col in ohe.columns:
        ohe.loc[0, action_col] = 1
    return pd.concat([scaled_features.reset_index(drop=True), ohe], axis=1)


# ===========================================================================
# Background Inference Worker
# ===========================================================================

def _run_inference_worker(
    buffer:            deque,
    buffer_lock:       threading.Lock,
    inference_trigger: threading.Event,
    state_lock:        threading.Lock,
    shared_state:      dict,           # keys: "action", "quality"
    shutdown_event:    threading.Event,
) -> None:

    logging.info("[Worker] Background inference thread started.")

    while not shutdown_event.is_set():

        # ── Wait for the main thread to signal that a new inference is due ──
        # Timeout of 1 s ensures the thread wakes up to check shutdown_event
        # even if the main thread never calls .set() (e.g. prolonged Idle).
        triggered = inference_trigger.wait(timeout=1.0)

        if shutdown_event.is_set():
            break

        if not triggered:
            # Timeout expired with no trigger — loop back and wait again.
            continue

        # ── Acknowledge the trigger immediately so the next .set() is clean ─
        inference_trigger.clear()

        # ── Step 1: Snapshot — hold buffer_lock for the minimum possible time.
        # list() on a deque of 90 dicts copies references only; it is O(N) but
        # N=90 and each element is a plain dict, so this completes in < 50 µs.
        with buffer_lock:
            if len(buffer) < buffer.maxlen:
                # Buffer not yet full — not enough data for quality stage.
                # Skip this trigger; the main thread will fire another one soon.
                continue
            action_snapshot  = list(buffer)[-ACTION_WINDOW_SIZE:]  # last 30
            quality_snapshot = list(buffer)                         # all 90

        # buffer_lock is now RELEASED.  All heavy work below runs lock-free.

        # ── Step 2a: Stage 1 — Fast Action Buffer (30 frames) ────────────────
        action_features_raw = build_window_features(action_snapshot)
        action_features_scaled = pd.DataFrame(
            scaler.transform(action_features_raw),
            columns=FEATURE_COLS,
        )
        pred_action = action_model.predict(action_features_scaled)[0]

        # ── Step 2b: Stage 2 — Deep Quality Buffer (90 frames) ───────────────
        quality_features_raw = build_window_features(quality_snapshot)
        quality_features_scaled = pd.DataFrame(
            scaler.transform(quality_features_raw),
            columns=FEATURE_COLS,
        )
        quality_input = build_quality_input(quality_features_scaled, pred_action)
        pred_quality  = float(quality_model.predict(quality_input)[0])

        # ── Step 3: Write results — hold state_lock for two assignments only ─
        with state_lock:
            shared_state["action"]  = pred_action
            shared_state["quality"] = pred_quality

    logging.info("[Worker] Background inference thread exiting.")


# ===========================================================================
# UI overlay
# ===========================================================================

def _draw_overlay(
    frame:   np.ndarray,
    status:  str,
    action:  str,
    quality: float,
    fps:     float,
) -> None:
    """Render the HUD panel on the frame in-place."""
    panel = frame.copy()
    cv2.rectangle(panel, (8, 8), (390, 120), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

    # Force quality to N/A when the system cannot make a meaningful judgement
    if status in ("Status: Idle", "Status: No Subject"):
        quality_str = "N/A"
    elif np.isnan(quality):
        quality_str = "--"
    else:
        quality_str = f"{quality:.2f}"

    cv2.putText(frame, status,
                (18, 34),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Action : {action}",
                (18, 62),  cv2.FONT_HERSHEY_SIMPLEX, 0.68, (50, 255, 80),  2, cv2.LINE_AA)
    cv2.putText(frame, f"Quality: {quality_str}",
                (18, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.68, (50, 255, 80),  2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:5.1f}",
                (18, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)


# ===========================================================================
# Utilities
# ===========================================================================

def _parse_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _resolve_backend(name: str, source) -> int | None:
    if not name:
        return None
    name = name.lower()
    if name == "auto":
        name = "ffmpeg" if (isinstance(source, str) and "://" in source) else "dshow"
    backend = BACKEND_MAP.get(name)
    if backend is None and name not in ("any", "dshow"):
        logging.warning("Backend '%s' unavailable — falling back to default.", name)
    return backend


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="V3 Real-time Action & Form Inference")
    parser.add_argument("--source",     default="1",
                        help="Webcam index or DroidCam/RTSP URL  (default: 1)")
    parser.add_argument("--width",      type=int, default=None)
    parser.add_argument("--height",     type=int, default=None)
    parser.add_argument("--buffer",     type=int, default=MAIN_BUFFER_MAXLEN,
                        help=f"Main deque length  (default: {MAIN_BUFFER_MAXLEN})")
    parser.add_argument("--idle-thresh", type=float,
                        default=IDLE_RANGE_THRESHOLD_DEFAULT,
                        help=(
                            "Maximum single-key peak-to-peak range (px) below which "
                            "the subject is considered idle. "
                            f"Default {IDLE_RANGE_THRESHOLD_DEFAULT} px is calibrated for "
                            "pixel-space coords at 640×480. "
                            "Use ~60 for 1280×720; use 0.05 for normalised [0,1] coords."
                        ))
    parser.add_argument("--ema-alpha",  type=float,
                        default=EMA_ALPHA_DEFAULT,
                        help=(
                            "EMA smoothing factor for the quality score display (0–1). "
                            f"Default {EMA_ALPHA_DEFAULT}. "
                            "Higher → faster response / more jitter. "
                            "Lower  → smoother / more lag."
                        ))
    parser.add_argument("--backend",    default="auto",
                        choices=["auto", "any", "msmf", "dshow", "ffmpeg"])
    args = parser.parse_args()

    idle_range_threshold: float = args.idle_thresh
    ema_alpha:            float = args.ema_alpha

    logging.info(
        "Models: action(%d feat) | quality(%d feat) | "
        "idle-thresh=%.1f px | ema-alpha=%.2f",
        len(FEATURE_COLS),
        len(FEATURE_COLS) + len(ACTION_OHE_COLS),
        idle_range_threshold,
        ema_alpha,
    )

    # ── Camera ──────────────────────────────────────────────────────────────
    source  = _parse_source(args.source)
    backend = _resolve_backend(args.backend, source)
    capture = VideoCaptureThread(
        source, width=args.width, height=args.height, backend=backend
    ).start()

    # =========================================================================
    # 1. Shared State & Synchronisation
    # =========================================================================

    # ── Main buffer (90-frame rolling window of raw per-frame feature dicts) ─
    buffer: deque[dict] = deque(maxlen=args.buffer)

    # Protects ONLY the deque above.  Never held for more than a list() copy.
    buffer_lock = threading.Lock()

    # Set by the main thread every 5th Running frame to wake the worker.
    # Cleared by the worker immediately on wakeup.
    inference_trigger = threading.Event()

    # Protects ONLY shared_state below.  Never held for more than two reads/writes.
    state_lock = threading.Lock()

    # The single source of truth for the latest ML predictions.
    # Written exclusively by the worker thread; read exclusively by the main thread.
    shared_state: dict = {"action": "--", "quality": np.nan}

    # Signals the worker to exit cleanly on shutdown.
    shutdown_event = threading.Event()

    # =========================================================================
    # 2. Background Worker Thread
    # =========================================================================
    worker = threading.Thread(
        target=_run_inference_worker,
        kwargs=dict(
            buffer=buffer,
            buffer_lock=buffer_lock,
            inference_trigger=inference_trigger,
            state_lock=state_lock,
            shared_state=shared_state,
            shutdown_event=shutdown_event,
        ),
        daemon=True,
        name="InferenceWorker",
    )
    worker.start()

    # ── MediaPipe Pose ───────────────────────────────────────────────────────
    mp_pose = mp_solutions.pose
    mp_draw = mp_drawing
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Frame-level state (main thread only — no lock needed) ────────────────
    frame_counter: int   = 0
    status:        str   = "Status: Warming"
    fps_start             = time.perf_counter()
    fps_count:     int   = 0
    current_fps:   float = 0.0

    # EMA quality: interpolated toward the latest worker prediction every frame.
    # Lives entirely on the main thread — never shared, never needs a lock.
    ema_quality:   float = np.nan

    logging.info("UI loop started — press 'q' to quit.")

    # =========================================================================
    # 3. Main UI Thread Loop
    # =========================================================================
    try:
        while True:

            # ── Pull latest frame from the camera queue ──────────────────────
            frame = capture.read()
            if frame is None:
                continue

            frame_counter += 1
            fps_count     += 1
            now = time.perf_counter()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count   = 0
                fps_start   = now

            frame_h, frame_w = frame.shape[:2]

            # ── Run MediaPipe (main thread — unavoidable; it owns the GL context)
            frame_rgb               = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results                 = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # ── Read latest raw prediction from worker (< 1 µs critical section)
            with state_lock:
                display_action = shared_state["action"]
                raw_quality    = shared_state["quality"]

            # ── EMA blend — runs every frame, no lock needed ──────────────────
            # Interpolates ema_quality toward raw_quality at rate ema_alpha.
            # Between worker fires (every 5 frames) raw_quality is stale, so
            # ema_quality smoothly converges to the last known target.
            # When the worker writes a new raw_quality, ema_quality smoothly
            # pivots toward it over the next several frames.
            if not np.isnan(raw_quality):
                if np.isnan(ema_quality):
                    ema_quality = raw_quality            # cold start: snap to first value
                else:
                    ema_quality = ema_alpha * raw_quality + (1.0 - ema_alpha) * ema_quality

            # =================================================================
            # Heuristic Gatekeeper
            # =================================================================

            # ── Check 1: Visibility ───────────────────────────────────────────
            if not results.pose_landmarks:
                status = "Status: No Subject"

                # Append a zero-row to preserve timeline alignment in the buffer
                with buffer_lock:
                    buffer.append(empty_feature_row())

                # Quality forced to "N/A" inside _draw_overlay based on status
                _draw_overlay(frame, status, display_action, ema_quality, current_fps)
                cv2.imshow("Action & Form Evaluation [V3]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue  # ← skip all ML work; go straight to next frame

            # ── Subject visible: draw skeleton ────────────────────────────────
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(50, 255, 80),   thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=2),
            )

            row = extract_pose_features(
                results.pose_landmarks.landmark,
                frame_width=frame_w,
                frame_height=frame_h,
            )

            # ── Append to buffer, snapshot 5-frame idle window atomically ────
            # Holding buffer_lock only for the append + shallow 5-element copy.
            with buffer_lock:
                buffer.append(row)
                buffer_len   = len(buffer)
                idle_snapshot = (
                    list(buffer)[-IDLE_WINDOW_SIZE:]
                    if buffer_len >= IDLE_WINDOW_SIZE
                    else None
                )

            # ── Check 2: Idle State ───────────────────────────────────────────
            if (
                idle_snapshot is not None
                and compute_motion_range(idle_snapshot) < idle_range_threshold
            ):
                status = "Status: Idle"
                # Quality forced to "N/A" inside _draw_overlay based on status
                _draw_overlay(frame, status, display_action, ema_quality, current_fps)
                cv2.imshow("Action & Form Evaluation [V3]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue  # ← skip ML inference; go straight to next frame

            # ── Warming guard ─────────────────────────────────────────────────
            if buffer_len < buffer.maxlen:
                status = "Status: Warming"
                _draw_overlay(frame, status, display_action, ema_quality, current_fps)
                cv2.imshow("Action & Form Evaluation [V3]", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ── Running — gatekeeper passed ───────────────────────────────────
            status = "Status: Running"

            # =================================================================
            # 4. Inference Throttle
            # Every 5th frame: wake the worker.  Zero ML math on this thread.
            # =================================================================
            if frame_counter % INFERENCE_EVERY_N_FRAMES == 0:
                inference_trigger.set()   # non-blocking; worker wakes asynchronously

            # =================================================================
            # UI Overlay — always rendered with the last known predictions.
            # The worker updates shared_state independently; the display here
            # uses whatever was read at the top of this loop iteration.
            # =================================================================
            _draw_overlay(frame, status, display_action, ema_quality, current_fps)
            cv2.imshow("Action & Form Evaluation [V3]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # ── Clean shutdown ───────────────────────────────────────────────────
        # Signal the worker first so inference_trigger.wait() unblocks.
        shutdown_event.set()
        inference_trigger.set()   # unblock any pending .wait() immediately

        worker.join(timeout=2.0)
        if worker.is_alive():
            logging.warning("Worker thread did not exit cleanly within 2 s.")

        capture.stop()
        pose.close()
        cv2.destroyAllWindows()
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    main()