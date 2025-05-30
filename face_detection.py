import cv2
from typing import Union
import numpy as np

from tinyface import FaceEmbedder, Face
from utils import s_print

from tqdm import tqdm

# https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html

detector = cv2.FaceDetectorYN.create(
    "./data/face_detection_yunet_2023mar.onnx", "", (0, 0), score_threshold=0.5
)

embedder = FaceEmbedder()


def detect_faces(img_input: Union[str, np.ndarray], silent=False) -> list[Face]:
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise ValueError(f"Could not read image from path: {img_input}")
    elif isinstance(img_input, np.ndarray):
        img = img_input

    height, width, _ = img.shape
    detector.setInputSize((width, height))

    s_print(" - Detecting faces", silent)

    _, faces = detector.detect(img)
    res = []

    s_print(" - Processing detected faces", silent)

    for face in tqdm(faces, disable=silent):
        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])

        face_landmark_5 = np.array(
            [
                [face[4], face[5]],
                [face[6], face[7]],
                [face[8], face[9]],
                [face[10], face[11]],
                [face[12], face[13]],
            ]
        )

        embedding, normed_embedding = embedder.calc_embedding(img, face_landmark_5)

        res.append(
            Face(
                bounding_box=[x, y, w, h],
                score=face[14],
                landmark_5=face_landmark_5,
                embedding=embedding,
                normed_embedding=normed_embedding,
            ),
        )
    s_print("", silent)
    return res
