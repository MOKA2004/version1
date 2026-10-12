import cv2
import time
from insightface.app import FaceAnalysis

print("Loading InsightFace with liveness...")

app = FaceAnalysis(
    name="buffalo_l",
    addons=["liveness"],
    liveness_mode="normal",
    liveness_threshold=0.8
)

app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

print("Model loaded.")
print("Starting camera...")

cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("ERROR: Could not open /dev/video0")
    exit()

time.sleep(2)

for i in range(30):

    ret, frame = cap.read()

    if not ret:
        print("Frame capture failed")
        continue

    faces = app.get(frame)

    print(f"\nFrame {i + 1}: {len(faces)} face(s)")

    for face in faces:

        liveness = getattr(face, "liveness", None)

        print("Liveness object:", liveness)

        if liveness is not None:
            print("Status:", getattr(liveness, "status", None))
            print("Is Live:", getattr(liveness, "is_live", None))
            print("Live Score:", getattr(liveness, "live_score", None))

cap.release()

print("\nTest completed.")