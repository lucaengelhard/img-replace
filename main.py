from face_detection import detect_faces, frame_faces
from tinyface import FacePair, TinyFace
import cv2

tinyface = TinyFace()
tinyface.prepare()

faces = detect_faces("./imgs/base_img.jpg")
shifted_faces = [faces[-1]] + faces[:-1]

pairs = []


# output_img = tinyface.swap_faces(
#     cv2.imread("./imgs/base_img.jpg"),
#     face_pairs=[
#         FacePair(reference=faces[i]["cutout"], destination=shifted_faces[i]["cutout"])
#         for i in range(len(faces))
#     ],
# )
