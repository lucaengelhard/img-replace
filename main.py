import cv2

import argparse

from face_detection import detect_faces
from face_replacement import Modified_TinyFace
from face_creation import read_db, create_db
from face_blur import blur_faces
from utils import display_imgs, get_image_paths, parse_path

# Setup
tinyface = Modified_TinyFace()
tinyface.prepare()

default_test_path = "./imgs/base_img.jpg"


# Arguments
parser = argparse.ArgumentParser(
    prog="Face Replace", description="Detect and Replace faces in images"
)

## Positional Arguments
parser.add_argument("mode", nargs="?", default="replace")

## Flag Arguments
parser.add_argument("-f", "--filename", help="Input file name")
parser.add_argument("-d", "--directory", help="Input directory name")
parser.add_argument("-o", "--output", help="Output dir path")
parser.add_argument(
    "-db",
    "--database",
    help="Database path of generated faces",
    default="faces_db.json",
)
parser.add_argument(
    "-a", "--amount", type=int, help="Amount of created faces", default=10
)
# TODO: parser.add_argument("-ov", "--overwrite", type=bool, help="Overwrite?", default=False)

## Parse Arguments
args = parser.parse_args()


def replace():
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


def create():
    create_db(args.amount, args.database)


def blur():
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

    res = []
    for path in file_paths:
        print()
        print(path)
        # Read Image
        img = cv2.imread(path)

        # Detect Faces
        faces = detect_faces(img)

        # Blur Faces
        output_img = blur_faces(img, faces)

        if not args.output == None:
            output_path = output_folder / (path.stem + "_swapped" + path.suffix)

            cv2.imwrite(output_path, output_img)
        else:
            res.append([output_img, faces])

    if args.output == None:
        display_imgs([r[0] for r in res], faces=[f[1] for f in res])


if args.mode == None or args.mode == "replace":
    replace()
elif args.mode == "create":
    create()
elif args.mode == "blur":
    blur()
