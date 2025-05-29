from tinyface.typing import (
    Distance,
    FaceLandmark5,
    Points,
)

from tinyface import Face

import cv2
import numpy


def distance_to_face_landmark_5(points: Points, distance: Distance) -> FaceLandmark5:
    x = points[:, 0::2] + distance[:, 0::2]
    y = points[:, 1::2] + distance[:, 1::2]

    face_landmark_5 = numpy.stack((x, y), axis=-1)
    return face_landmark_5


def frame_faces(img, faces: list[Face], features=False):
    for face in faces:
        x = int(face.bounding_box[0])
        y = int(face.bounding_box[1])
        w = int(face.bounding_box[2])
        h = int(face.bounding_box[3])
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1,
        )

        if features:
            for landmark in face.landmark_5:
                cv2.circle(
                    img,
                    (numpy.uint(landmark[0]), numpy.uint(landmark[1])),
                    2,
                    (255, 0, 0),
                    1,
                ),
