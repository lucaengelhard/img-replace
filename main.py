import cv2

import argparse

from face_detection import detect_faces, detect_faces_threads
from face_replacement import Modified_TinyFace
from face_creation import read_db, create_db
from face_blur import blur_faces
from utils import display_imgs, frame_faces, get_arg_paths, conditional_call

# Setup
tinyface = Modified_TinyFace()
tinyface.prepare()


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
parser.add_argument("-ov", "--overwrite", action="store_true", help="Overwrite?")
parser.add_argument(
    "-nm", "--no-multithreading", action="store_true", help="Disable multithreading"
)

## Parse Arguments
args = parser.parse_args()

multithreading = not args.no_multithreading

print(multithreading)


def replace():
    file_paths, output_folder = get_arg_paths(args)

    db = read_db(args.database)

    res = []
    for path in file_paths:
        print()
        print(path)
        # Read Image
        img = cv2.imread(path)

        # Detect Faces
        faces = conditional_call(
            detect_faces_threads, detect_faces, multithreading, img
        )

        # Replace Faces
        output_img = conditional_call(
            tinyface.swap_faces_db_threads,
            tinyface.swap_faces_db,
            multithreading,
            img,
            faces,
            db,
        )

        if not args.output == None:
            output_path = output_folder / (path.stem + "_swapped" + path.suffix)
            cv2.imwrite(output_path, output_img)

        res.append([output_img, faces])

    if args.output == None:
        display_imgs([r[0] for r in res], faces=[f[1] for f in res])

    return res


def create():
    create_db(args.amount, args.database)


def blur():
    file_paths, output_folder = get_arg_paths(args)

    res = []
    for path in file_paths:
        print()
        print(path)
        # Read Image
        img = cv2.imread(path)

        # Detect Faces
        faces = conditional_call(
            detect_faces_threads, detect_faces, multithreading, img
        )

        # Blur Faces
        output_img = blur_faces(img, faces)

        if not args.output == None:
            output_path = output_folder / (path.stem + "_swapped" + path.suffix)
            cv2.imwrite(output_path, output_img)

        res.append([output_img, faces])

    if args.output == None:
        display_imgs([r[0] for r in res], faces=[f[1] for f in res])

    return res


def detect():
    file_paths, output_folder = get_arg_paths(args)

    res = []
    for path in file_paths:
        print()
        print(path)
        # Read Image
        img = cv2.imread(path)

        # Detect Faces
        faces = conditional_call(
            detect_faces_threads, detect_faces, multithreading, img
        )

        if not args.output == None:
            output_path = output_folder / (path.stem + "_swapped" + path.suffix)
            cv2.imwrite(output_path, frame_faces(img, faces, True))

        res.append([img, faces])

    if args.output == None:
        display_imgs([r[0] for r in res], faces=[f[1] for f in res])

    return res


if args.mode == None or args.mode == "replace":
    replace()
elif args.mode == "create":
    create()
elif args.mode == "blur":
    blur()
elif args.mode == "detect":
    detect()
