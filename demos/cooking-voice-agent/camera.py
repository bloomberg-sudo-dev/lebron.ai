"""Webcam capture using OpenCV. Returns JPEG bytes or None if unavailable."""

from typing import Optional

MAX_WIDTH = 1280


class Camera:
    def __init__(self, device_index: int = 0):
        self._cap = None
        self._device_index = device_index
        self.available = False

    def __enter__(self):
        try:
            import cv2
            self._cv2 = cv2
            self._cap = cv2.VideoCapture(self._device_index)
            self.available = self._cap.isOpened()
        except ImportError:
            print("[Camera] opencv-python not installed — running in audio-only mode.")
        return self

    def __exit__(self, *args):
        if self._cap and self._cap.isOpened():
            self._cap.release()

    def capture_frame(self) -> Optional[bytes]:
        if not self.available or self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            scale = MAX_WIDTH / w
            new_w, new_h = int(w * scale), int(h * scale)
            frame = self._cv2.resize(frame, (new_w, new_h))
        _, buffer = self._cv2.imencode(".jpg", frame, [self._cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
