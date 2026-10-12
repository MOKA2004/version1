import cv2
from insightface.app import FaceAnalysis

print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l"
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded successfully.")