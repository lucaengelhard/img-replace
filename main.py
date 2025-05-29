import sys
import os

import cv2
from tinyface import TinyFace

from face_detection import detect_faces
from utils import frame_faces

# Setup
tinyface = TinyFace()
tinyface.prepare()

# Arguments
path = "./imgs/base_img.jpg"
if len(sys.argv) > 1:
    path = sys.argv[1]


img = cv2.imread(path)

# tiny = tinyface.get_many_faces(img)

faces = detect_faces(path)
# shifted_faces = [faces[-1]] + faces[:-1]

# output_img = tinyface.swap_faces(
#     img,
#     face_pairs=[
#         FacePair(reference=faces[i], destination=shifted_faces[i])
#         for i in range(len(faces))
#     ],
# )

frame_faces(img, faces, True)

cv2.imshow("res", img)
cv2.waitKey(0)
