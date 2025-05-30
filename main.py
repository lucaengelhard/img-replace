import cv2
from tqdm import tqdm

import argparse
from pathlib import Path, PurePath
import os

from face_detection import detect_faces
from face_replacement import Modified_TinyFace
from utils import frame_faces, scale_image_to, get_image_paths, parse_path

# Setup
tinyface = Modified_TinyFace()
tinyface.prepare()


# Arguments
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--filename", help="Input file name")
parser.add_argument("-d", "--directory", help="Input directory name")
parser.add_argument("-o", "--output", help="Output dir path")

args = parser.parse_args()

default_test_path = "./imgs/base_img.jpg"
file_paths = [parse_path(default_test_path)]

if not args.filename == None or not args.directory == None:
    file_paths.clear()
    if not args.filename == None:
        file_paths.append(parse_path(args.filename))

    if not args.directory == None:
        file_paths.extend(get_image_paths(parse_path(args.directory)))

if not args.output == None:
    output_folder = parse_path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)


for path in tqdm(file_paths):
    # Read Image
    img = cv2.imread(path)

    # Detect Faces
    faces = detect_faces(img)

    # Replace Faces
    output_img = tinyface.swap_faces_db(img, faces, faces)

    if not args.output == None:
        output_path = output_folder / (path.stem + "_swapped" + path.suffix)

        cv2.imwrite(output_path, output_img)


# Display result
# cv2.imshow("res", scale_image_to(frame_faces(output_img, faces), 1920, 1080))
# cv2.waitKey(0)
