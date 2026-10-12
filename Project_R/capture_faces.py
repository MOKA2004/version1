import cv2
import os


# -----------------------------
# Get employee ID
# -----------------------------

employee_id = input("Enter employee ID: ").strip()

if not employee_id:
    print("Employee ID cannot be empty.")
    exit()


# -----------------------------
# Create employee folder
# -----------------------------

folder_path = os.path.join("data", "faces", employee_id)

os.makedirs(folder_path, exist_ok=True)


# -----------------------------
# Load face detector
# -----------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# -----------------------------
# Open camera
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera.")
    exit()


# -----------------------------
# Capture face samples
# -----------------------------

sample_count = 0
frame_count = 0

print("\nLook at the camera.")
print("Move your face slightly while samples are being captured.")
print("Press 'q' to stop.\n")


while sample_count < 30:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # Process the largest detected face
    if len(faces) > 0:

        x, y, width, height = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        # Crop the face
        face_image = gray[
            y:y + height,
            x:x + width
        ]

        # Save every few frames
        if frame_count % 5 == 0:

            sample_count += 1

            filename = os.path.join(
                folder_path,
                f"face_{sample_count:02d}.jpg"
            )

            cv2.imwrite(filename, face_image)

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        # Show progress
        cv2.putText(
            frame,
            f"Samples: {sample_count}/30",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Employee Face Enrollment", frame)

    frame_count += 1

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Release resources
# -----------------------------

camera.release()
cv2.destroyAllWindows()


print(f"\nEnrollment completed.")
print(f"Employee ID: {employee_id}")
print(f"Samples captured: {sample_count}")
print(f"Saved in: {folder_path}")