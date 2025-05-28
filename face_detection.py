import cv2
import matplotlib.pyplot as plt


detector = cv2.FaceDetectorYN.create(
    "./data/face_detection_yunet_2023mar.onnx", "", (0, 0), score_threshold=0.5
)


def detect_faces(path: str):
    img = cv2.imread(path)
    height, width, _ = img.shape
    detector.setInputSize((width, height))
    _, faces = detector.detect(img)
    res = []

    for face in faces:
        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])
        res.append(
            {"x": x, "y": y, "w": w, "h": h, "cutout": img[y : y + h, x : x + w]}
        )

    return res


def frame_faces(path):
    img = cv2.imread(path)
    for face in detect_faces(path):
        cv2.rectangle(
            img,
            (face["x"], face["y"]),
            (face["x"] + face["w"], face["y"] + face["h"]),
            (0, 255, 0),
            4,
        )

    cv2.imshow("framed", img)
    cv2.waitKey(0)
