import cv2

for camera_index in range(4):
    print(f"\nTesting camera {camera_index}...")

    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        print(f"Camera {camera_index}: NOT AVAILABLE")
        camera.release()
        continue

    print(f"Camera {camera_index}: AVAILABLE")
    print("Press Q to close this camera.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Could not read frame.")
            break

        cv2.putText(
            frame,
            f"CAMERA INDEX: {camera_index}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(f"Camera {camera_index}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
