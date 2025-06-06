import cv2
from typing import Union
import numpy as np

from tinyface import FaceEmbedder, Face
from src.utils import s_print, get_img

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html

detector = cv2.FaceDetectorYN.create(
    "./data/face_detection_yunet_2023mar.onnx", "", (0, 0), score_threshold=0.5
)

embedder = FaceEmbedder()


def _extract_landmarks(face_data: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [face_data[4], face_data[5]],
            [face_data[6], face_data[7]],
            [face_data[8], face_data[9]],
            [face_data[10], face_data[11]],
            [face_data[12], face_data[13]],
        ]
    )


def _process_face(face_data: np.ndarray, img: np.ndarray) -> Face:
    x, y, w, h = map(int, face_data[:4])
    landmarks = _extract_landmarks(face_data)
    embedding, normed_embedding = embedder.calc_embedding(img, landmarks)
    return Face(
        bounding_box=[x, y, w, h],
        score=face_data[14],
        landmark_5=landmarks,
        embedding=embedding,
        normed_embedding=normed_embedding,
    )


def detect_faces(img_input: Union[str, np.ndarray], silent=False) -> list[Face]:
    img = get_img(img_input)
    height, width, _ = img.shape

    s_print(silent, " - Detecting faces (Single Thread)")
    detector.setInputSize((width, height))

    _, faces = detector.detect(img)

    res = []
    if faces is None or len(faces) == 0:
        s_print(silent, " - No faces detected")
        return res

    s_print(silent, " - Processing detected faces")

    for face in tqdm(faces, disable=silent):
        res.append(_process_face(face, img))

    s_print(silent)
    return res


def detect_faces_threads(img_input: Union[str, np.ndarray], silent=False) -> list[Face]:
    img = get_img(img_input)
    height, width, _ = img.shape

    s_print(silent, " - Detecting faces")
    detector.setInputSize((width, height))

    _, faces = detector.detect(img)

    res = []
    if faces is None or len(faces) == 0:
        s_print(silent, " - No faces detected")
        return res

    s_print(silent, " - Processing detected faces")

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_process_face, face, img) for face in faces]
        for future in tqdm(as_completed(futures), total=len(futures), disable=silent):
            res.append(future.result())

    return res
