import sys

import cv2

from face_detection import detect_faces
from face_replacement import Modified_TinyFace
from utils import frame_faces, scale_image_to

# Setup
tinyface = Modified_TinyFace()
tinyface.prepare()

# Arguments
path = "./imgs/base_img.jpg"
if len(sys.argv) > 1:
    path = sys.argv[1]

# Read Image
img = cv2.imread(path)

# Detect Faces
faces = detect_faces(img)

# Replace Faces
output_img = tinyface.swap_faces_db(img, faces, faces)

# Display result
cv2.imshow("res", scale_image_to(frame_faces(output_img, faces), 1920, 1080))
cv2.waitKey(0)
