import time
from collections import deque

class PerfMonitor:
    def __init__(self, window=50):
        self.t = deque(maxlen=window)
    def start(self): self._s = time.perf_counter()
    def lap(self):
        dt = (time.perf_counter() - self._s) * 1000
        self.t.append(dt)
    @property
    def fps(self):
        return 1000 / (sum(self.t) / len(self.t)) if self.t else 0
