import cv2
import os
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

FACES_DIRECTORY = "data/faces"
MODEL_FILE = "face_model.yml"
LABEL_FILE = "employee_labels.txt"

FACE_SIZE = (200, 200)


# ==========================================
# CHECK OPENCV
# ==========================================

if not hasattr(cv2, "face"):
    print("ERROR: OpenCV face module is not available.")
    print("Install opencv-contrib-python.")
    exit()


# ==========================================
# CHECK FACE DIRECTORY
# ==========================================

if not os.path.exists(FACES_DIRECTORY):
    print("ERROR: Face data directory does not exist.")
    print(f"Expected: {FACES_DIRECTORY}")
    exit()


# ==========================================
# CREATE LBPH RECOGNIZER
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()


# ==========================================
# STORAGE
# ==========================================

face_images = []
face_labels = []

employee_ids = []


# ==========================================
# READ EMPLOYEE FOLDERS
# ==========================================

for employee_id in sorted(os.listdir(FACES_DIRECTORY)):

    employee_path = os.path.join(
        FACES_DIRECTORY,
        employee_id
    )

    if not os.path.isdir(employee_path):
        continue

    print()
    print(f"Loading employee: {employee_id}")

    employee_ids.append(employee_id)

    image_files = sorted(
        os.listdir(employee_path)
    )

    employee_image_count = 0

    for image_name in image_files:

        image_path = os.path.join(
            employee_path,
            image_name
        )

        # Only process image files
        if not image_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(
                f"Could not read: {image_path}"
            )
            continue

        # Normalize image
        image = cv2.equalizeHist(image)

        # Resize every training image
        image = cv2.resize(
            image,
            FACE_SIZE
        )

        face_images.append(image)

        face_labels.append(employee_id)

        employee_image_count += 1

    print(
        f"Loaded {employee_image_count} images "
        f"for employee {employee_id}"
    )


# ==========================================
# CHECK TRAINING DATA
# ==========================================

if len(face_images) == 0:

    print()
    print("ERROR: No face images found.")
    exit()


if len(employee_ids) < 2:

    print()
    print("WARNING:")
    print("Only one employee is registered.")
    print(
        "Unknown-person rejection will be "
        "less reliable with only one employee."
    )


# ==========================================
# CREATE NUMERIC LABELS
# ==========================================

unique_employee_ids = sorted(
    set(face_labels)
)

name_to_label = {
    employee_id: index
    for index, employee_id
    in enumerate(unique_employee_ids)
}


numeric_labels = np.array(
    [
        name_to_label[employee_id]
        for employee_id in face_labels
    ],
    dtype=np.int32
)


# ==========================================
# TRAIN MODEL
# ==========================================

print()
print("========================================")
print("Training LBPH face recognition model")
print("========================================")

print(
    f"Training images : {len(face_images)}"
)

print(
    f"Employees        : {len(unique_employee_ids)}"
)

print(
    f"Face size        : {FACE_SIZE}"
)

recognizer.train(
    face_images,
    numeric_labels
)


# ==========================================
# SAVE MODEL
# ==========================================

recognizer.write(
    MODEL_FILE
)


# ==========================================
# SAVE LABEL MAPPING
# ==========================================

with open(
    LABEL_FILE,
    "w"
) as file:

    for employee_id in unique_employee_ids:

        label = name_to_label[
            employee_id
        ]

        file.write(
            f"{label},{employee_id}\n"
        )


# ==========================================
# DISPLAY RESULTS
# ==========================================

print()
print("========================================")
print("TRAINING COMPLETED")
print("========================================")

print()
print("Employee mappings:")

for employee_id in unique_employee_ids:

    label = name_to_label[
        employee_id
    ]

    print(
        f"Label {label} -> Employee {employee_id}"
    )

print()
print(f"Model saved: {MODEL_FILE}")
print(f"Labels saved: {LABEL_FILE}")

print()
print("Phase 4 training completed.")