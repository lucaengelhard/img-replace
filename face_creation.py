import requests
import cv2
import numpy as np
import json
from tqdm import tqdm

from tinyface import Face
from face_detection import detect_faces

url = "https://thispersondoesnotexist.com/"


def get_new_face():
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return max(detect_faces(img, silent=True), key=lambda face: face.score)


def create_db(amount=10, filename="faces_db.json"):
    faces_data = []
    print(f"Creating {amount} faces")

    for _ in tqdm(range(amount)):
        face = get_new_face()

        data = {
            "bounding_box": face.bounding_box,
            "score": float(face.score),
            "landmark_5": face.landmark_5.tolist(),
            "embedding": face.embedding.tolist(),
            "normed_embedding": face.normed_embedding.tolist(),
        }

        faces_data.append(data)

    with open(filename, "w") as f:
        json.dump(faces_data, f)

    print(f"Faces saved to {filename}")


def read_db(filename="faces_db.json"):
    with open(filename, "r") as f:
        faces_data = json.load(f)

    faces_list = []

    for face_dict in faces_data:
        face = Face(
            bounding_box=face_dict["bounding_box"],
            score=np.float32(face_dict["score"]),
            landmark_5=np.array(face_dict["landmark_5"], dtype="float32"),
            embedding=np.array(face_dict["embedding"], dtype="float32"),
            normed_embedding=np.array(face_dict["normed_embedding"], dtype="float32"),
        )

        faces_list.append(face)

    return faces_list
