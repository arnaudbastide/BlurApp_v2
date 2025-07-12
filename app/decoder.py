import cv2

class Decoder:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError("Cannot open video")
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame
        self.cap.release()
