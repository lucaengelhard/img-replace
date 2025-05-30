import cv2

import argparse

from face_detection import detect_faces
from face_replacement import Modified_TinyFace
from face_creation import read_db
from utils import display_imgs, get_image_paths, parse_path

# Setup
tinyface = Modified_TinyFace()
tinyface.prepare()


# Arguments
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--filename", help="Input file name")
parser.add_argument("-d", "--directory", help="Input directory name")
parser.add_argument("-o", "--output", help="Output dir path")
parser.add_argument(
    "-db",
    "--database",
    help="Database path of generated faces",
    default="faces_db.json",
)

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

db = read_db(args.database)

res = []
for path in file_paths:
    print()
    print(path)
    # Read Image
    img = cv2.imread(path)

    # Detect Faces
    faces = detect_faces(img)

    # Replace Faces
    output_img = tinyface.swap_faces_db(img, faces, db)

    if not args.output == None:
        output_path = output_folder / (path.stem + "_swapped" + path.suffix)

        cv2.imwrite(output_path, output_img)
    else:
        res.append([output_img, faces])


if args.output == None:
    display_imgs([r[0] for r in res], faces=[f[1] for f in res])
