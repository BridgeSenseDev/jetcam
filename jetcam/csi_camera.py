import cv2
import numpy as np
import threading
import traitlets
import atexit


class CSICamera(traitlets.HasTraits):
    value = traitlets.Any()
    width = traitlets.Integer(default_value=224)
    height = traitlets.Integer(default_value=224)
    fps = traitlets.Integer(default_value=30)
    capture_width = traitlets.Integer(default_value=1920)
    capture_height = traitlets.Integer(default_value=1080)
    running = traitlets.Bool(default_value=False)

    def __init__(self, *args, **kwargs):
        super(CSICamera, self).__init__(*args, **kwargs)
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._running = False
        self.cap = None
        self._init_camera()
        atexit.register(self.stop)

    def _gstreamer_pipeline(self, flip_method=0):
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width=(int){self.capture_width}, height=(int){self.capture_height}, "
            f"format=(string)NV12, framerate=(fraction){self.fps}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){self.width}, height=(int){self.height}, format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        )

    def _init_camera(self):
        self.cap = cv2.VideoCapture(self._gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")

    def _read(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return cv2.resize(frame, (self.width, self.height))
            else:
                raise RuntimeError("Failed to capture frame")
        else:
            raise RuntimeError("Camera is not opened")

    def read(self):
        if self._running:
            raise RuntimeError("Cannot read directly while camera is running")
        return self._read()

    def _capture_frames(self):
        while self._running:
            self.value = self._read()

    @traitlets.observe("running")
    def _on_running(self, change):
        if change["new"] and not change["old"]:
            self._running = True
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()
        elif change["old"] and not change["new"]:
            self._running = False
            if hasattr(self, "thread"):
                self.thread.join()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
