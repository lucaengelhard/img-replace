import requests
import cv2
import numpy as np
import json
import os
from tqdm import tqdm

from tinyface import Face
from src.core.face_detection import detect_faces
from src.utils import parse_path
import src.defaults as defaults

url = "https://thispersondoesnotexist.com/"


def get_new_face():
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return max(detect_faces(img, silent=True), key=lambda face: face.score)


def create_db(amount=defaults.DATABASE_SIZE, filename="faces_db.json"):
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


def read_db(filename=defaults.DATABASE):
    if filename == None:
        filename = defaults.DATABASE

    filename = parse_path(filename)

    if not os.path.exists(filename):
        print(" - No Database found")

        yes = {"y", "yes"}
        no = {"n", "no"}

        answer = input(" - Create new database (internet connection required) [y/n]: ")

        if answer.lower() in yes:
            cont = False
            amount_int = defaults.DATABASE_SIZE
            while not cont:
                amount_str = input(" - How big should the new database be? ")

                if not amount_str.isdigit():
                    print("   The amount has to be a number!")
                    continue

                amount_int = int(amount_str)

                if amount_int > 1:
                    cont = True
                    continue

                print("   The amount has to be at least 1!")

            create_db(amount_int, filename)

        elif answer.lower() in no:
            return None

    print(f"Reading db {filename}")

    with open(filename, "r") as f:
        faces_data = json.load(f)

    faces_list = []

    for face_dict in tqdm(faces_data):
        face = Face(
            bounding_box=face_dict["bounding_box"],
            score=np.float32(face_dict["score"]),
            landmark_5=np.array(face_dict["landmark_5"], dtype="float32"),
            embedding=np.array(face_dict["embedding"], dtype="float32"),
            normed_embedding=np.array(face_dict["normed_embedding"], dtype="float32"),
        )

        faces_list.append(face)

    return faces_list
