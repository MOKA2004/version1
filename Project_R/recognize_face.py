import cv2
from collections import deque, Counter

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_FILE = "face_model.yml"
LABEL_FILE = "employee_labels.txt"

FACE_SIZE = (200, 200)

# Start with 70 because your current
# enrolled employees were producing
# scores around 53 and 63.
RECOGNITION_THRESHOLD = 70

# Number of recent predictions to remember
HISTORY_SIZE = 5

# Employee must appear at least this many
# times in recent predictions.
REQUIRED_MATCHES = 3


# ==========================================
# CHECK OPENCV
# ==========================================

if not hasattr(cv2, "face"):

    print(
        "ERROR: OpenCV face module is not available."
    )

    print(
        "Install opencv-contrib-python."
    )

    exit()


# ==========================================
# LOAD MODEL
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

try:

    recognizer.read(
        MODEL_FILE
    )

except Exception as error:

    print("Could not load face model.")
    print(error)

    exit()


# ==========================================
# LOAD EMPLOYEE LABELS
# ==========================================

employee_labels = {}

try:

    with open(
        LABEL_FILE,
        "r"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            label, employee_id = line.split(",")

            employee_labels[
                int(label)
            ] = employee_id

except Exception as error:

    print("Could not load employee labels.")
    print(error)

    exit()


print()
print("Registered employees:")

for label, employee_id in employee_labels.items():

    print(
        f"Label {label} -> Employee {employee_id}"
    )


# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


if face_detector.empty():

    print(
        "ERROR: Could not load Haar Cascade."
    )

    exit()


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    exit()


# ==========================================
# RECOGNITION HISTORY
#
# Each detected face gets its own history
# based on position.
# ==========================================

recognition_history = {}


# ==========================================
# FUNCTION:
# CREATE FACE KEY
# ==========================================

def get_face_key(x, y, width, height):

    center_x = x + width // 2
    center_y = y + height // 2

    # Divide the image into rough regions.
    # This prevents tiny movements from
    # creating completely new identities.

    key_x = center_x // 100
    key_y = center_y // 100

    return (
        key_x,
        key_y
    )


# ==========================================
# FUNCTION:
# PROCESS FACE
# ==========================================

def process_face(face):

    # Convert to histogram-normalized face
    face = cv2.equalizeHist(face)

    # Resize to exactly the same size
    # used during training.

    face = cv2.resize(
        face,
        FACE_SIZE
    )

    return face


# ==========================================
# START
# ==========================================

print()
print("========================================")
print("FACE RECOGNITION STARTED")
print("========================================")

print(
    f"Recognition threshold: "
    f"{RECOGNITION_THRESHOLD}"
)

print(
    f"Required matches: "
    f"{REQUIRED_MATCHES}/{HISTORY_SIZE}"
)

print()
print("Press 'q' to quit.")
print()


# ==========================================
# MAIN CAMERA LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "Could not read camera frame."
        )

        break


    # --------------------------------------
    # GRAYSCALE
    # --------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------
    # FACE DETECTION
    # --------------------------------------

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    # --------------------------------------
    # PROCESS EACH FACE
    # --------------------------------------

    for (
        x,
        y,
        width,
        height
    ) in faces:

        # Extract face
        face = gray[
            y:y + height,
            x:x + width
        ]


        # Normalize face
        processed_face = process_face(
            face
        )


        # ----------------------------------
        # LBPH PREDICTION
        # ----------------------------------

        label, distance = recognizer.predict(
            processed_face
        )


        # ----------------------------------
        # GET EMPLOYEE ID
        # ----------------------------------

        predicted_employee = (
            employee_labels.get(
                label
            )
        )


        # ----------------------------------
        # INITIAL DECISION
        # ----------------------------------

        if (
            predicted_employee is not None
            and distance <= RECOGNITION_THRESHOLD
        ):

            current_prediction = (
                predicted_employee
            )

        else:

            current_prediction = "Unknown"


        # ----------------------------------
        # FACE TRACKING KEY
        # ----------------------------------

        face_key = get_face_key(
            x,
            y,
            width,
            height
        )


        if face_key not in recognition_history:

            recognition_history[
                face_key
            ] = deque(
                maxlen=HISTORY_SIZE
            )


        # Add current prediction
        recognition_history[
            face_key
        ].append(
            current_prediction
        )


        # ----------------------------------
        # STABLE RECOGNITION
        # ----------------------------------

        recent_predictions = (
            recognition_history[
                face_key
            ]
        )


        prediction_counts = Counter(
            recent_predictions
        )


        most_common_prediction, count = (
            prediction_counts.most_common(1)[0]
        )


        # ----------------------------------
        # FINAL DECISION
        # ----------------------------------

        if (
            most_common_prediction != "Unknown"
            and count >= REQUIRED_MATCHES
        ):

            final_employee = (
                most_common_prediction
            )

            status = "Recognized"

            box_color = (
                0,
                255,
                0
            )

            display_text = (
                f"{final_employee} | "
                f"Score: {distance:.1f}"
            )

        else:

            final_employee = None

            status = "Unknown"

            box_color = (
                0,
                0,
                255
            )

            display_text = (
                f"Unknown | "
                f"Score: {distance:.1f}"
            )


        # ----------------------------------
        # DRAW FACE BOX
        # ----------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (
                x + width,
                y + height
            ),
            box_color,
            2
        )


        # ----------------------------------
        # DISPLAY RESULT
        # ----------------------------------

        cv2.putText(
            frame,
            display_text,
            (
                x,
                max(y - 10, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2
        )


    # --------------------------------------
    # DISPLAY CAMERA
    # --------------------------------------

    cv2.imshow(
        "Employee Face Recognition",
        frame
    )


    # --------------------------------------
    # QUIT
    # --------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()

print()
print("Face recognition stopped.")