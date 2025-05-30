from tinyface import Face

import cv2
import numpy


def frame_faces(img, faces: list[Face], features=False, scale=1):
    newImg = img.copy()
    for face in faces:
        x = int(face.bounding_box[0])
        y = int(face.bounding_box[1])
        w = int(face.bounding_box[2])
        h = int(face.bounding_box[3])
        cv2.rectangle(
            newImg,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2 * scale,
        )

        if features:
            for landmark in face.landmark_5:
                cv2.circle(
                    newImg,
                    (numpy.uint(landmark[0]), numpy.uint(landmark[1])),
                    2 * scale,
                    (255, 0, 0),
                    1 * scale,
                ),
    return newImg


def scale_image_to(img, max_w, max_h):
    # Get original dimensions
    height, width = img.shape[:2]

    #   Calculate the scaling factor
    scale_w = max_w / width
    scale_h = max_h / height
    scale = min(scale_w, scale_h)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
