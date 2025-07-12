import cv2

class Encoder:
    def __init__(self, path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.wr = cv2.VideoWriter(path, fourcc, fps, size)

    def write(self, frame):
        self.wr.write(frame)

    def close(self):
        self.wr.release()
