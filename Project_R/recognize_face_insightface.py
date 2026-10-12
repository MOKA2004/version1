import cv2
import atexit
import os
import time
import pickle
import argparse
import threading
import numpy as np
import onnxruntime as ort
from urllib.parse import urlsplit

from collections import defaultdict, deque
from insightface.app import FaceAnalysis

from attendance import record_entry, record_exit
from camera_manager import get_camera, list_cameras, open_camera

# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="InsightFace Employee Attendance Recognition"
)

parser.add_argument(
    "--camera",
    required=False,
    help="Direct device index, device path, or network camera URL (legacy override)",
)

parser.add_argument(
    "--camera-id",
    required=False,
    help="Configured camera ID. If omitted, all enabled cameras assigned to --mode are used.",
)

parser.add_argument(
    "--mode",
    required=True,
    choices=["ENTRY", "EXIT"],
    help="Fixed camera responsibility"
)

args = parser.parse_args()

CAMERA_MODE = args.mode


def configured_cameras():
    if args.camera:
        return [{
            "id": "direct-source",
            "name": "Direct camera source",
            "source": args.camera.strip(),
            "source_type": "ip" if args.camera.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")) else "webcam",
            "roles": [CAMERA_MODE],
            "enabled": True,
        }]

    if args.camera_id:
        camera = get_camera(args.camera_id)
        if camera is None:
            raise ValueError(f"Configured camera '{args.camera_id}' was not found.")
        if not camera["enabled"]:
            raise ValueError(f"Camera '{camera['name']}' is disabled.")
        if CAMERA_MODE not in camera["roles"]:
            raise ValueError(f"Camera '{camera['name']}' is not assigned to {CAMERA_MODE}.")
        return [camera]

    return [
        camera for camera in list_cameras(enabled_only=True, include_source=True)
        if CAMERA_MODE in camera["roles"]
    ]


class MultiCameraSource:
    """Read frames in round-robin order through the common camera interface."""

    def __init__(self, cameras):
        self.dynamic = not bool(args.camera)
        self.camera_id = args.camera_id
        self.sources = []
        self.index = 0
        self.frame_counts = []
        self.last_source_index = 0
        self.current_camera_name = ""
        self.current_camera_id = ""
        self._signature = None
        self._last_refresh = 0.0
        self.refresh(force=True, initial=cameras)

    def refresh(self, *, force=False, initial=None):
        if not self.dynamic and not force:
            return
        now = time.monotonic()
        if not force and now - self._last_refresh < 2.0:
            return
        self._last_refresh = now

        if initial is not None:
            cameras = initial
        elif self.camera_id:
            selected = get_camera(self.camera_id)
            cameras = [selected] if selected and selected["enabled"] and CAMERA_MODE in selected["roles"] else []
        else:
            cameras = [
                camera for camera in list_cameras(enabled_only=True, include_source=True)
                if CAMERA_MODE in camera["roles"]
            ]

        signature = tuple(
            (camera["id"], camera["name"], camera["source_type"], camera.get("source"), tuple(camera["roles"]))
            for camera in cameras
        )
        if signature == self._signature:
            return

        for _, old_source in self.sources:
            old_source.release()
        self.sources = []
        self.frame_counts = []
        self.index = 0
        self._signature = signature

        for camera in cameras:
            source = open_camera(camera)
            if source.isOpened():
                self.sources.append((camera, source))
                self.frame_counts.append(0)
            else:
                source.release()
                print(f"Unable to open camera: {camera['name']}")
        self.cameras = cameras

    def isOpened(self):
        self.refresh()
        return bool(self.sources)

    def read(self):
        self.refresh()
        if not self.sources:
            return False, None
        for _ in range(len(self.sources)):
            source_index = self.index
            camera_info, source = self.sources[source_index]
            self.index = (self.index + 1) % len(self.sources)
            success, frame = source.read_latest()
            if success and frame is not None:
                self.last_source_index = source_index
                self.current_camera_name = camera_info["name"]
                self.current_camera_id = camera_info["id"]
                self.frame_counts[source_index] += 1
                return True, frame
        return False, None

    def should_process_current_frame(self):
        return self.frame_counts[self.last_source_index] % PROCESS_EVERY_N_FRAMES == 0

    def release(self):
        for _, source in self.sources:
            source.release()

    def active_cameras(self):
        return [(camera["id"], camera["name"]) for camera, _ in self.sources]


CAMERAS = configured_cameras()
CAMERA_LABEL = ", ".join(camera["name"] for camera in CAMERAS) or "No assigned cameras"
CAMERA_DEVICE = CAMERA_LABEL


def display_camera_source(source):

    parsed = urlsplit(source)

    if parsed.scheme.lower() in {"rtsp", "rtsps"}:
        host = parsed.hostname or "unknown-host"
        try:
            port = f":{parsed.port}" if parsed.port else ""
        except ValueError:
            port = ""

        return f"{parsed.scheme.lower()}://{host}{port}/…"

    return source


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "embeddings",
    "face_embeddings.pkl"
)

# Recognition threshold
SIMILARITY_THRESHOLD = 0.45

# Minimum difference between best and second-best employee
CONFIDENCE_MARGIN = 0.05

def positive_env_int(name, default):
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        print(f"Invalid {name}; using default {default}.")
        return default


# More input detail helps the detector find smaller, more distant faces.
RECOGNITION_WIDTH = positive_env_int("FACE_RECOGNITION_WIDTH", 640)
DETECTION_SIZE = (RECOGNITION_WIDTH, RECOGNITION_WIDTH)


def normalized_env_float(name, default):
    try:
        return min(1.0, max(0.0, float(os.getenv(name, str(default)))))
    except ValueError:
        print(f"Invalid {name}; using default {default}.")
        return default


# Default scan region covers most of the camera view so several people can
# enter together. Adjust these fractions to match the actual doorway framing.
FACE_ROI_X1 = normalized_env_float("FACE_ROI_X1", 0.10)
FACE_ROI_Y1 = normalized_env_float("FACE_ROI_Y1", 0.10)
FACE_ROI_X2 = normalized_env_float("FACE_ROI_X2", 0.90)
FACE_ROI_Y2 = normalized_env_float("FACE_ROI_Y2", 1.0)
if FACE_ROI_X2 <= FACE_ROI_X1 or FACE_ROI_Y2 <= FACE_ROI_Y1:
    print("Invalid face ROI bounds; scanning the full frame instead.")
    FACE_ROI_X1, FACE_ROI_Y1, FACE_ROI_X2, FACE_ROI_Y2 = 0.0, 0.0, 1.0, 1.0

# The capture worker already drops stale frames, so do not deliberately skip
# additional frames before detection. Increase this only if CPU load is too high.
PROCESS_EVERY_N_FRAMES = positive_env_int("FACE_PROCESS_EVERY_N_FRAMES", 1)
FACE_MAX_NUM = positive_env_int("FACE_MAX_NUM", 5)
BOX_TRACK_WIDTH = positive_env_int("FACE_BOX_TRACK_WIDTH", 640)
BOX_OVERLAY_MAX_AGE = 0.75
ORT_INTRA_OP_THREADS = positive_env_int(
    "FACE_ORT_INTRA_OP_THREADS", min(4, os.cpu_count() or 1)
)

# Stable recognition
HISTORY_SIZE = 5
REQUIRED_MATCHES = min(
    HISTORY_SIZE,
    positive_env_int("FACE_REQUIRED_MATCHES", 2),
)

# Attendance protection
ATTENDANCE_COOLDOWN = 5

# Tracking
TRACK_DISTANCE_THRESHOLD = 120
TRACK_TIMEOUT = 2.0


# ============================================================
# LIVENESS / ANTI-SPOOFING
# ============================================================

LIVENESS_THRESHOLD = 0.80

# Number of live results required before allowing recognition
LIVENESS_HISTORY_SIZE = 3
LIVENESS_REQUIRED = 2


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

PREVIEW_WIDTH = 900
PREVIEW_HEIGHT = 506


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = f"Face Recognition - {CAMERA_MODE}"


# ============================================================
# TRACKING
# ============================================================

track_history = defaultdict(
    lambda: deque(maxlen=HISTORY_SIZE)
)

track_last_seen = {}

track_last_position = {}

attendance_cooldown = {}

# Liveness history for each tracked face
liveness_history = defaultdict(
    lambda: deque(maxlen=LIVENESS_HISTORY_SIZE)
)


# ============================================================
# EMBEDDINGS
# ============================================================

employee_embeddings = {}
employee_embedding_matrix = np.empty((0, 0), dtype=np.float32)
employee_embedding_row_groups = np.empty((0,), dtype=np.int32)
employee_embedding_group_ids = []

embeddings_modified_time = 0


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(force=False):

    global employee_embeddings
    global employee_embedding_matrix
    global employee_embedding_row_groups
    global employee_embedding_group_ids
    global embeddings_modified_time

    if not os.path.exists(EMBEDDINGS_FILE):

        print()
        print("ERROR: Embeddings file not found:")
        print(EMBEDDINGS_FILE)
        print()

        return False

    try:

        modified_time = os.path.getmtime(
            EMBEDDINGS_FILE
        )

        if (
            not force
            and
            modified_time == embeddings_modified_time
        ):
            return True

        with open(
            EMBEDDINGS_FILE,
            "rb"
        ) as file:

            loaded_embeddings = pickle.load(
                file
            )

        if not isinstance(
            loaded_embeddings,
            dict
        ):

            print(
                "ERROR: Invalid embeddings format."
            )

            return False

        new_embeddings = {}
        normalized_rows = []
        row_group_ids = []
        employee_ids = []
        employee_group_indexes = {}

        for employee_id, embeddings in loaded_embeddings.items():

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32
            )

            if embeddings.ndim == 1:

                embeddings = embeddings.reshape(
                    1,
                    -1
                )

            if embeddings.size == 0:
                continue

            if embeddings.ndim != 2:
                continue

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            valid_rows = np.isfinite(norms[:, 0]) & (norms[:, 0] > 0)
            embeddings = embeddings[valid_rows]
            norms = norms[valid_rows]
            if embeddings.size == 0:
                continue

            if normalized_rows and embeddings.shape[1] != normalized_rows[0].shape[1]:
                print(f"Skipping embeddings for {employee_id}: dimension mismatch.")
                continue

            employee_id = str(employee_id)
            if employee_id not in employee_group_indexes:
                employee_group_indexes[employee_id] = len(employee_ids)
                employee_ids.append(employee_id)
            group_index = employee_group_indexes[employee_id]
            normalized_rows.append(embeddings / norms)
            row_group_ids.extend([group_index] * len(embeddings))

            new_embeddings[
                str(employee_id)
            ] = embeddings

        employee_embeddings = new_embeddings
        if normalized_rows:
            employee_embedding_matrix = np.concatenate(normalized_rows, axis=0).astype(np.float32, copy=False)
            employee_embedding_row_groups = np.asarray(row_group_ids, dtype=np.int32)
            employee_embedding_group_ids = employee_ids
        else:
            employee_embedding_matrix = np.empty((0, 0), dtype=np.float32)
            employee_embedding_row_groups = np.empty((0,), dtype=np.int32)
            employee_embedding_group_ids = []

        embeddings_modified_time = modified_time

        print()
        print(
            "Employee embeddings loaded:"
        )
        print(
            "---------------------------"
        )

        for employee_id, embeddings in employee_embeddings.items():

            print(
                f"Employee {employee_id}: "
                f"{embeddings.shape}"
            )

        print()

        return True

    except Exception as e:

        print(
            "ERROR loading embeddings:",
            e
        )

        return False


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    embedding_a,
    embedding_b
):

    denominator = (
        np.linalg.norm(embedding_a)
        *
        np.linalg.norm(embedding_b)
    )

    if denominator == 0:

        return 0.0

    return float(
        np.dot(
            embedding_a,
            embedding_b
        )
        /
        denominator
    )


# ============================================================
# FIND EMPLOYEE
# ============================================================

def find_employee(face_embedding):

    if employee_embedding_matrix.size == 0:

        return None, 0.0, 0.0

    query = np.asarray(face_embedding, dtype=np.float32).reshape(-1)
    if query.size != employee_embedding_matrix.shape[1]:
        return None, 0.0, 0.0

    query_norm = np.linalg.norm(query)
    if not np.isfinite(query_norm) or query_norm == 0:
        return None, 0.0, 0.0

    similarities = employee_embedding_matrix @ (query / query_norm)
    group_scores = np.full(len(employee_embedding_group_ids), -1.0, dtype=np.float32)
    np.maximum.at(group_scores, employee_embedding_row_groups, similarities)
    ranked_indices = np.argsort(group_scores)[::-1]

    best_index = ranked_indices[0]
    best_employee = employee_embedding_group_ids[best_index]

    best_score = float(group_scores[best_index])

    if len(ranked_indices) > 1:

        second_best_index = ranked_indices[1]
        second_best_score = float(group_scores[second_best_index])

    else:

        second_best_index = None
        second_best_score = 0.0

    confidence_margin = (
        best_score
        -
        second_best_score
    )

    if best_score < SIMILARITY_THRESHOLD:

        return (
            None,
            best_score,
            confidence_margin
        )

    if (
        len(ranked_indices) > 1
        and
        confidence_margin < CONFIDENCE_MARGIN
    ):

        print(
            f"AMBIGUOUS FACE | "
            f"Best={best_employee} "
            f"{best_score:.3f} | "
            f"Second={employee_embedding_group_ids[second_best_index]} "
            f"{second_best_score:.3f} | "
            f"Margin={confidence_margin:.3f}"
        )

        return (
            None,
            best_score,
            confidence_margin
        )

    return (
        best_employee,
        best_score,
        confidence_margin
    )


# ============================================================
# RECORD ATTENDANCE
# ============================================================

def record_attendance_event(employee_id):

    current_time = time.time()

    cooldown_key = (
        str(employee_id),
        CAMERA_MODE
    )

    # --------------------------------------------------------
    # COOLDOWN
    # --------------------------------------------------------

    if cooldown_key in attendance_cooldown:

        elapsed = (
            current_time
            -
            attendance_cooldown[
                cooldown_key
            ]
        )

        if elapsed < ATTENDANCE_COOLDOWN:

            return

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    if CAMERA_MODE == "ENTRY":

        result = record_entry(
            str(employee_id)
        )

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    else:

        result = record_exit(
            str(employee_id)
        )

    attendance_cooldown[
        cooldown_key
    ] = current_time

    print()
    print(
        "========================================"
    )

    print(
        f"[{CAMERA_MODE}] "
        f"Employee {employee_id}"
    )

    print(result)

    print(
        "========================================"
    )


# ============================================================
# TRACK ID
# ============================================================

def get_track_id(
    center_x,
    center_y,
    camera_id="",
):

    current_time = time.time()

    current_position = (
        center_x,
        center_y
    )
    track_prefix = f"{camera_id}:" if camera_id else ""

    # --------------------------------------------------------
    # Remove expired tracks
    # --------------------------------------------------------

    expired_tracks = []

    for track_id, last_seen in track_last_seen.items():

        if (
            current_time
            -
            last_seen
            >
            TRACK_TIMEOUT
        ):

            expired_tracks.append(
                track_id
            )

    for track_id in expired_tracks:

        track_last_seen.pop(
            track_id,
            None
        )

        track_last_position.pop(
            track_id,
            None
        )

        track_history.pop(
            track_id,
            None
        )

        liveness_history.pop(
            track_id,
            None
        )

    # --------------------------------------------------------
    # Find nearest existing track
    # --------------------------------------------------------

    best_track = None

    best_distance = float("inf")

    for track_id, position in track_last_position.items():

        if track_prefix and not track_id.startswith(track_prefix):
            continue

        distance = np.linalg.norm(
            np.array(current_position)
            -
            np.array(position)
        )

        if (
            distance < best_distance
            and
            distance < TRACK_DISTANCE_THRESHOLD
        ):

            best_distance = distance
            best_track = track_id

    # --------------------------------------------------------
    # Existing track
    # --------------------------------------------------------

    if best_track is not None:

        track_id = best_track

    # --------------------------------------------------------
    # New track
    # --------------------------------------------------------

    else:

        track_id = (
            f"{track_prefix}"
            f"{current_time:.6f}_"
            f"{center_x}_"
            f"{center_y}"
        )

    track_last_seen[
        track_id
    ] = current_time

    track_last_position[
        track_id
    ] = current_position

    return track_id


# ============================================================
# LIVENESS CHECK
# ============================================================

def check_liveness(
    face,
    track_id
):

    liveness = getattr(
        face,
        "liveness",
        None
    )

    # --------------------------------------------------------
    # No liveness information
    # --------------------------------------------------------

    if liveness is None:

        print(
            "LIVENESS ERROR | "
            "No liveness information"
        )

        liveness_history[
            track_id
        ].clear()

        return (
            False,
            0.0,
            "LIVENESS ERROR"
        )

    # --------------------------------------------------------
    # InsightFace liveness result
    # --------------------------------------------------------

    if isinstance(
        liveness,
        dict
    ):

        status = liveness.get(
            "status"
        )

        is_live = bool(
            liveness.get(
                "is_live",
                False
            )
        )

        live_score = float(
            liveness.get(
                "live_score",
                0.0
            )
        )

    else:

        status = getattr(
            liveness,
            "status",
            None
        )

        is_live = bool(
            getattr(
                liveness,
                "is_live",
                False
            )
        )

        live_score = float(
            getattr(
                liveness,
                "live_score",
                0.0
            )
        )

    # --------------------------------------------------------
    # Invalid/rejected input
    # --------------------------------------------------------

    if status != "ok":

        liveness_history[
            track_id
        ].clear()

        return (
            False,
            live_score,
            "SPOOF / REJECTED"
        )

    # --------------------------------------------------------
    # Score-based live decision
    # --------------------------------------------------------

    live_result = (
        is_live
        and
        live_score >= LIVENESS_THRESHOLD
    )

    liveness_history[
        track_id
    ].append(
        live_result
    )

    live_count = sum(
        liveness_history[
            track_id
        ]
    )

    # --------------------------------------------------------
    # Require stable liveness
    # --------------------------------------------------------

    if live_count >= LIVENESS_REQUIRED:

        return (
            True,
            live_score,
            "LIVE"
        )

    # --------------------------------------------------------
    # Not yet stable
    # --------------------------------------------------------

    if not live_result:

        return (
            False,
            live_score,
            "SPOOF"
        )

    return (
        False,
        live_score,
        "VERIFYING LIVE"
    )


# ============================================================
# DRAW BOX AND LABEL
# ============================================================

def draw_face_box(
    display_frame,
    x1,
    y1,
    x2,
    y2,
    color,
    label
):

    cv2.rectangle(
        display_frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    label_height = 30

    label_y1 = max(
        0,
        y1 - label_height
    )

    cv2.rectangle(
        display_frame,
        (x1, label_y1),
        (x2, y1),
        color,
        -1
    )

    cv2.putText(
        display_frame,
        label,
        (
            x1 + 5,
            max(
                20,
                y1 - 8
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            255,
            255
        ),
        2
    )


# ============================================================
# PROCESS FACES
# ============================================================

def process_faces(
    display_frame,
    faces,
    scale_x,
    scale_y,
    camera_id="",
    offset_x=0,
    offset_y=0,
):
    annotations = []

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        # ----------------------------------------------------
        # Convert recognition coordinates
        # ----------------------------------------------------

        x1 = int(x1 * scale_x) + offset_x

        y1 = int(y1 * scale_y) + offset_y

        x2 = int(x2 * scale_x) + offset_x

        y2 = int(y2 * scale_y) + offset_y

        # ----------------------------------------------------
        # Keep inside frame
        # ----------------------------------------------------

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            display_frame.shape[1] - 1,
            x2
        )

        y2 = min(
            display_frame.shape[0] - 1,
            y2
        )

        # ----------------------------------------------------
        # Center
        # ----------------------------------------------------

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        track_id = get_track_id(
            center_x,
            center_y,
            camera_id,
        )

        # ====================================================
        # LIVENESS CHECK
        # ====================================================

        is_live, live_score, live_status = check_liveness(
            face,
            track_id
        )

        # ====================================================
        # SPOOF / NOT VERIFIED
        # ====================================================

        if not is_live:

            # Absolutely clear recognition history.
            # A spoof must never inherit previous recognition.
            track_history[
                track_id
            ].clear()

            # ------------------------------------------------
            # RED BOX
            # ------------------------------------------------

            color = (
                0,
                0,
                255
            )

            if live_status == "VERIFYING LIVE":

                label = (
                    f"VERIFYING "
                    f"{live_score:.2f}"
                )

            elif live_status == "LIVENESS ERROR":

                label = "LIVENESS ERROR"

            else:

                label = (
                    f"SPOOF "
                    f"{live_score:.2f}"
                )

            annotations.append((x1, y1, x2, y2, color, label))

            # Do NOT access face.embedding.
            # Do NOT recognize.
            # Do NOT record attendance.

            continue

        # ====================================================
        # REAL PERSON
        # ====================================================

        # At this point liveness has passed.
        # Recognition can safely continue.

        if not hasattr(
            face,
            "embedding"
        ):

            annotations.append((x1, y1, x2, y2, (0, 0, 255), "NO EMBEDDING"))

            track_history[
                track_id
            ].clear()

            continue

        # ----------------------------------------------------
        # Recognition
        # ----------------------------------------------------

        employee_id, score, margin = find_employee(
            face.embedding
        )

        # ====================================================
        # UNKNOWN / AMBIGUOUS
        # ====================================================

        if employee_id is None:

            color = (
                0,
                0,
                255
            )

            label = (
                f"UNKNOWN "
                f"{score:.2f}"
            )

            # Unknown face must not build history.
            track_history[
                track_id
            ].clear()

        # ====================================================
        # KNOWN REAL EMPLOYEE
        # ====================================================

        else:

            color = (
                0,
                255,
                0
            )

            label = (
                f"{employee_id} "
                f"{score:.2f} "
                f"LIVE"
            )

            # ------------------------------------------------
            # Add recognition result
            # ------------------------------------------------

            track_history[
                track_id
            ].append(
                str(employee_id)
            )

            history = track_history[
                track_id
            ]

            # ------------------------------------------------
            # Count matches
            # ------------------------------------------------

            counts = {}

            for recognized_id in history:

                counts[
                    recognized_id
                ] = (
                    counts.get(
                        recognized_id,
                        0
                    )
                    +
                    1
                )

            best_id = max(
                counts,
                key=counts.get
            )

            best_count = counts[
                best_id
            ]

            # ------------------------------------------------
            # Stable recognition
            # ------------------------------------------------

            if best_count >= REQUIRED_MATCHES:

                record_attendance_event(
                    best_id
                )

                track_history[
                    track_id
                ].clear()

        # ====================================================
        # DRAW BOX
        # ====================================================

        annotations.append((x1, y1, x2, y2, color, label))

    return annotations


class LatestFrameInferenceWorker:
    """Run face inference off the UI thread, keeping only each camera's newest pending frame."""

    def __init__(self, recognition_app):
        self.app = recognition_app
        self._condition = threading.Condition()
        self._pending_frames = {}
        self._ready_cameras = deque()
        self._overlays = {}
        self._overlay_lock = threading.Lock()
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run,
            name="face-recognition-worker",
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def submit(self, camera_id, camera_name, frame):
        with self._condition:
            if camera_id not in self._pending_frames:
                self._ready_cameras.append(camera_id)
            # Replace stale pending work instead of allowing latency to grow.
            self._pending_frames[camera_id] = (camera_name, frame)
            self._condition.notify()

    def get_annotations(self, camera_id, max_age=1.0):
        overlay = self.get_overlay(camera_id, max_age=max_age)
        return overlay[3] if overlay else ()

    def get_overlay(self, camera_id, max_age=BOX_OVERLAY_MAX_AGE):
        with self._overlay_lock:
            overlay = self._overlays.get(camera_id)
            if overlay is None:
                return None
            captured_at, published_at, frame_gray, frame_shape, annotations = overlay
            if time.monotonic() - published_at > max_age:
                return None
            return overlay

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self):
        inference_samples_ms = deque(maxlen=30)
        total_samples_ms = deque(maxlen=30)
        inference_passes = 0

        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stopping or bool(self._ready_cameras)
                )
                if self._stopping:
                    return
                camera_id = self._ready_cameras.popleft()
                camera_name, frame = self._pending_frames.pop(camera_id)

            total_started = time.perf_counter()
            try:
                captured_at = time.monotonic()
                frame_height, frame_width = frame.shape[:2]
                gray_width = min(BOX_TRACK_WIDTH, frame_width)
                gray_height = max(1, round(frame_height * gray_width / frame_width))
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.resize(
                    frame_gray,
                    (gray_width, gray_height),
                    interpolation=cv2.INTER_AREA,
                )
                roi_x1 = int(frame_width * FACE_ROI_X1)
                roi_y1 = int(frame_height * FACE_ROI_Y1)
                roi_x2 = min(frame_width, max(roi_x1 + 1, int(frame_width * FACE_ROI_X2)))
                roi_y2 = min(frame_height, max(roi_y1 + 1, int(frame_height * FACE_ROI_Y2)))
                roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                roi_height, roi_width = roi_frame.shape[:2]
                recognition_height = max(
                    1,
                    round(roi_height * RECOGNITION_WIDTH / roi_width),
                )
                scale_x = roi_width / RECOGNITION_WIDTH
                scale_y = roi_height / recognition_height
                recognition_frame = cv2.resize(
                    roi_frame,
                    (RECOGNITION_WIDTH, recognition_height),
                    interpolation=(
                        cv2.INTER_AREA
                        if roi_width > RECOGNITION_WIDTH
                        else cv2.INTER_LINEAR
                    ),
                )

                inference_started = time.perf_counter()
                faces = self.app.get(
                    recognition_frame,
                    max_num=FACE_MAX_NUM,
                )
                inference_ms = (time.perf_counter() - inference_started) * 1000
                annotations = process_faces(
                    frame,
                    faces,
                    scale_x,
                    scale_y,
                    camera_id,
                    roi_x1,
                    roi_y1,
                )

                with self._overlay_lock:
                    self._overlays[camera_id] = (
                        captured_at,
                        time.monotonic(),
                        frame_gray,
                        frame.shape[:2],
                        tuple(annotations),
                    )

                inference_samples_ms.append(inference_ms)
                total_samples_ms.append((time.perf_counter() - total_started) * 1000)
                inference_passes += 1
                if inference_passes % inference_samples_ms.maxlen == 0:
                    average_inference = sum(inference_samples_ms) / len(inference_samples_ms)
                    average_total = sum(total_samples_ms) / len(total_samples_ms)
                    print(
                        f"Inference average (last {len(inference_samples_ms)} passes): "
                        f"{average_inference:.1f} ms app.get, {average_total:.1f} ms total; "
                        f"camera: {camera_name}; faces: {len(faces)}"
                    )
            except Exception as error:
                print(f"Face processing error on {camera_name}: {error}")


class OpticalFlowBoxTracker:
    """Propagate face boxes on display frames between recognition passes."""

    def __init__(self):
        self._states = {}
        self._overlay_tokens = {}

    @staticmethod
    def _points_for_box(gray, box):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = max(0, min(gray.shape[1] - 1, x1))
        y1 = max(0, min(gray.shape[0] - 1, y1))
        x2 = max(x1 + 1, min(gray.shape[1], x2))
        y2 = max(y1 + 1, min(gray.shape[0], y2))
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return np.empty((0, 1, 2), dtype=np.float32)
        points = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=48,
            qualityLevel=0.01,
            minDistance=4,
            blockSize=5,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        points[:, 0, 0] += x1
        points[:, 0, 1] += y1
        return points.astype(np.float32)

    @staticmethod
    def _gray_for_frame(frame, shape):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != shape:
            gray = cv2.resize(
                gray,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return gray

    def update(self, camera_id, frame, overlay):
        if overlay is not None:
            captured_at, published_at, source_gray, source_shape, annotations = overlay
            token = (captured_at, id(source_gray))
            if self._overlay_tokens.get(camera_id) != token:
                scale_x = source_gray.shape[1] / source_shape[1]
                scale_y = source_gray.shape[0] / source_shape[0]
                boxes = []
                for x1, y1, x2, y2, color, label in annotations:
                    box = (
                        x1 * scale_x,
                        y1 * scale_y,
                        x2 * scale_x,
                        y2 * scale_y,
                    )
                    boxes.append({
                        "box": box,
                        "points": self._points_for_box(source_gray, box),
                        "color": color,
                        "label": label,
                    })
                self._states[camera_id] = {
                    "gray": source_gray,
                    "boxes": boxes,
                    "published_at": published_at,
                }
                self._overlay_tokens[camera_id] = token

        state = self._states.get(camera_id)
        if state is None or not state["boxes"]:
            return ()
        if time.monotonic() - state["published_at"] > BOX_OVERLAY_MAX_AGE:
            self._states.pop(camera_id, None)
            self._overlay_tokens.pop(camera_id, None)
            return ()

        previous_gray = state["gray"]
        current_gray = self._gray_for_frame(frame, previous_gray.shape)
        rendered = []
        updated_boxes = []

        for item in state["boxes"]:
            points = item["points"]
            if points is None or len(points) < 3:
                continue

            next_points, status, errors = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_points is None or status is None:
                continue
            valid = status.reshape(-1).astype(bool)
            if errors is not None:
                valid &= errors.reshape(-1) < 40.0
            if np.count_nonzero(valid) < 3:
                continue

            old_valid = points.reshape(-1, 2)[valid]
            new_valid = next_points.reshape(-1, 2)[valid]
            displacement = np.median(new_valid - old_valid, axis=0)
            x1, y1, x2, y2 = item["box"]
            dx, dy = float(displacement[0]), float(displacement[1])
            box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

            # Reject implausible optical-flow jumps rather than leaving a box
            # detached from the detected face.
            if abs(dx) > previous_gray.shape[1] * 0.15 or abs(dy) > previous_gray.shape[0] * 0.15:
                continue
            if box[2] <= 0 or box[3] <= 0 or box[0] >= current_gray.shape[1] or box[1] >= current_gray.shape[0]:
                continue

            tracked_points = new_valid.reshape(-1, 1, 2).astype(np.float32)
            if len(tracked_points) < 12:
                replenished = self._points_for_box(current_gray, box)
                if len(replenished):
                    tracked_points = np.concatenate((tracked_points, replenished), axis=0)
            updated_boxes.append({**item, "box": box, "points": tracked_points})

            frame_scale_x = frame.shape[1] / current_gray.shape[1]
            frame_scale_y = frame.shape[0] / current_gray.shape[0]
            bx1, by1, bx2, by2 = box
            rendered.append((
                max(0, int(round(bx1 * frame_scale_x))),
                max(0, int(round(by1 * frame_scale_y))),
                min(frame.shape[1] - 1, int(round(bx2 * frame_scale_x))),
                min(frame.shape[0] - 1, int(round(by2 * frame_scale_y))),
                item["color"],
                item["label"],
            ))

        state["gray"] = current_gray
        state["boxes"] = updated_boxes
        return tuple(rendered)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "======================================"
    )

    print(
        "   FACE ATTENDANCE RECOGNITION"
    )

    print(
        "======================================"
    )

    print()

    print(
        f"Camera : {CAMERA_DEVICE}"
    )

    print(
        f"Mode   : {CAMERA_MODE}"
    )

    print(
        f"Embeddings : {EMBEDDINGS_FILE}"
    )

    print(
        f"Similarity threshold : "
        f"{SIMILARITY_THRESHOLD}"
    )

    print(
        f"Confidence margin : "
        f"{CONFIDENCE_MARGIN}"
    )

    print(
        f"Liveness threshold : "
        f"{LIVENESS_THRESHOLD}"
    )

    print()

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    if not load_embeddings(
        force=True
    ):

        return

    # ========================================================
    # LOAD INSIGHTFACE WITH LIVENESS
    # ========================================================

    print(
        "Loading InsightFace model "
        "with liveness..."
    )

    # Bound per-session CPU fan-out. The detector, recognizer and liveness
    # sessions run serially per face; allowing every session to claim all cores
    # can cause oversubscription on CPU-only machines.
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = ORT_INTRA_OP_THREADS
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    app = FaceAnalysis(
        name="buffalo_l",
        # Attendance uses only detection and embeddings. Skip unused age and
        # landmark models to reduce startup cost and memory use.
        allowed_modules=["detection", "recognition"],
        sess_options=session_options,
        addons=[
            "liveness"
        ],
        liveness_mode="normal",
        liveness_threshold=LIVENESS_THRESHOLD
    )

    app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE
    )

    print(
        "InsightFace model with "
        "liveness loaded."
    )
    print(
        f"ONNX Runtime CPU threads: {ORT_INTRA_OP_THREADS}; "
        f"recognition confirmations: {REQUIRED_MATCHES}"
    )

    print()

    # ========================================================
    # OPEN ASSIGNED CAMERAS
    # ========================================================

    if not CAMERAS:
        print(f"ERROR: No enabled cameras are assigned to {CAMERA_MODE}.")
        print("Configure cameras and assign their Entry/Exit role in the dashboard.")
        return

    camera = MultiCameraSource(CAMERAS)
    atexit.register(cv2.destroyAllWindows)
    atexit.register(camera.release)

    if not camera.isOpened():

        print()
        print(
            "ERROR: Could not open camera:"
        )

        print(
            display_camera_source(CAMERA_DEVICE)
        )

        return

    inference_worker = LatestFrameInferenceWorker(app)
    inference_worker.start()
    atexit.register(inference_worker.stop)
    box_tracker = OpticalFlowBoxTracker()

    # ========================================================
    # CAMERA SETTINGS
    # ========================================================

    print(
        f"{CAMERA_MODE} camera started."
    )

    print(
        f"Device: {display_camera_source(CAMERA_DEVICE)}"
    )

    print(
        "Liveness protection: ENABLED"
    )

    print(
        f"Maximum faces per pass: {FACE_MAX_NUM}"
    )

    print(
        "Press Q to quit."
    )

    print()

    # ========================================================
    # VARIABLES
    # ========================================================

    preview_windows = {}
    resolution_logged = False

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        try:
            ret, frame = camera.read()
        except cv2.error as error:
            print("Camera read error:", error)
            ret, frame = False, None

        active_cameras = dict(camera.active_cameras())
        active_ids = set(active_cameras)

        for camera_id in list(preview_windows):
            if camera_id not in active_ids:
                cv2.destroyWindow(preview_windows.pop(camera_id))

        for camera_id, camera_name in active_cameras.items():
            if camera_id not in preview_windows:
                window_name = f"{WINDOW_NAME} - {camera_name} [{camera_id[:8]}]"
                cv2.namedWindow(
                    window_name,
                    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                )
                cv2.resizeWindow(
                    window_name,
                    PREVIEW_WIDTH,
                    PREVIEW_HEIGHT
                )
                preview_windows[camera_id] = window_name

        if not ret:
            cv2.waitKey(1)
            time.sleep(0.002)
            continue

        if not resolution_logged:
            print(f"Camera frame resolution: {frame.shape[1]}x{frame.shape[0]}")
            print(
                "Door scan ROI: "
                f"x={FACE_ROI_X1:.2f}-{FACE_ROI_X2:.2f}, "
                f"y={FACE_ROI_Y1:.2f}-{FACE_ROI_Y2:.2f}"
            )
            resolution_logged = True

        camera_id = camera.current_camera_id
        camera_name = camera.current_camera_name
        if camera.should_process_current_frame():
            # CameraSource owns the captured array after read; the UI draws on
            # a separate preview copy while inference consumes this frame.
            inference_worker.submit(camera_id, camera_name, frame)

        display_frame = frame.copy()

        # ====================================================
        # CAMERA LABEL
        # ====================================================

        roi_x1 = int(display_frame.shape[1] * FACE_ROI_X1)
        roi_y1 = int(display_frame.shape[0] * FACE_ROI_Y1)
        roi_x2 = int(display_frame.shape[1] * FACE_ROI_X2)
        roi_y2 = int(display_frame.shape[0] * FACE_ROI_Y2)
        cv2.rectangle(display_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 215, 255), 2)

        cv2.putText(
            display_frame,
            f"{CAMERA_MODE} CAMERA - DOOR SCAN AREA",
            (
                20,
                35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (
                255,
                255,
                255
            ),
            2
        )

        cv2.putText(
            display_frame,
            "LIVENESS: ON",
            (
                20,
                65
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                255,
                255,
                255
            ),
            2
        )

        cv2.putText(
            display_frame,
            camera_name,
            (
                20,
                95
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                255,
                255,
                255
            ),
            2
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        overlay = inference_worker.get_overlay(camera_id)
        annotations = box_tracker.update(camera_id, frame, overlay)
        for annotation in annotations:
            draw_face_box(display_frame, *annotation)

        window_name = preview_windows.get(camera_id)
        if window_name:
            cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(
            1
        ) & 0xFF

        if key == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    inference_worker.stop()
    camera.release()

    cv2.destroyAllWindows()

    print()

    print(
        f"{CAMERA_MODE} camera stopped."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()