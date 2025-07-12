from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=10, n_init=2)

    def update(self, detections, frame):
        dets = [([x1,y1,x2-x1,y2-y1], conf, 0) for (x1,y1,x2,y2), conf in zip(*detections)]
        return self.tracker.update_tracks(dets, frame=frame)
