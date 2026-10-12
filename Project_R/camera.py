import argparse
import os

import cv2

from camera_manager import CameraSource


def main() -> None:
    parser = argparse.ArgumentParser(description="View a low-latency camera stream")
    parser.add_argument(
        "--source",
        default=os.getenv("HIKVISION_RTSP_URL", "0"),
        help="Camera device index or Hikvision RTSP URL (or set HIKVISION_RTSP_URL)",
    )
    parser.add_argument(
        "--transport",
        choices=("udp", "tcp"),
        default=os.getenv("FACE_RTSP_TRANSPORT", "udp").lower(),
        help="RTSP transport; UDP is lower-latency on a reliable LAN, TCP is more resilient",
    )
    args = parser.parse_args()
    os.environ.setdefault("FACE_RTSP_TRANSPORT", args.transport)

    camera = CameraSource(args.source)
    if not camera.isOpened():
        camera.release()
        raise SystemExit("Could not open camera; verify the source URL and credentials.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                continue
            cv2.imshow("Employee Attendance Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()