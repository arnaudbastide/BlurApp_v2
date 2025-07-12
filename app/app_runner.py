import cv2
from infra.logger import setup_logging
from core.models.multi_detector import MultiDetector
from .decoder import Decoder
from .encoder import Encoder
from .tracker import Tracker
from .blur_processor import BlurProcessor

class AppRunner:
    def __init__(self, cfg):
        setup_logging()
        self.cfg = cfg
        self.det = MultiDetector(cfg.face_model, cfg.person_model, cfg.plate_model, cfg.providers)
        self.dec = Decoder(cfg.video)
        self.enc = Encoder(cfg.output, self.dec.fps, (self.dec.w, self.dec.h)) if cfg.output else None
        self.trk = Tracker()
        self.blur = BlurProcessor(cfg)

    def run(self):
        for idx, frame in enumerate(self.dec):
            if idx % self.cfg.skip_frames != 0:
                continue
            dets = self.det(frame)
            tracks = self.trk.update(dets["persons"], frame)

            blurred = frame.copy()
            blurred = self.blur.blur(blurred, dets["faces"][0])
            blurred = self.blur.blur(blurred, dets["plates"][0])

            if self.enc:
                self.enc.write(blurred)
            cv2.imshow("ONNX App", blurred)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if self.enc:
            self.enc.close()
        cv2.destroyAllWindows()
