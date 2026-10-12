import cv2


# Open the camera
camera = cv2.VideoCapture(0)


# Load the face detection model
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


while True:

    # Capture a frame from the camera
    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Draw a rectangle around every detected face
    for (x, y, width, height) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

    # Display the camera
    cv2.imshow("Face Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release camera
camera.release()

# Close OpenCV windows
cv2.destroyAllWindows()