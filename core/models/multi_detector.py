from .face_detector import FaceDetector
from .person_detector import PersonDetector
from .plate_detector import PlateDetector

class MultiDetector:
    def __init__(self, face_path, person_path, plate_path, providers):
        self.face  = FaceDetector(face_path, providers=providers)
        self.person= PersonDetector(person_path, providers=providers)
        self.plate = PlateDetector(plate_path, providers=providers)

    def __call__(self, img):
        return {
            "faces":   self.face.detect(img),
            "persons": self.person.detect(img),
            "plates":  self.plate.detect(img),
        }
