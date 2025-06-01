import cv2
import argparse
from typing import Union

from face_detection import detect_faces, detect_faces_threads
from face_replacement import Modified_TinyFace
from face_creation import read_db, create_db
from face_blur import blur_faces
from utils import display_imgs, frame_faces, get_arg_paths, conditional_call

# Setup
tinyface = Modified_TinyFace()
tinyface.prepare()


class FaceReplace:
    def __init__(
        self,
        mode: str,
        filename: Union[str, None],
        directory: Union[str, None],
        output: Union[str, None],
        database: Union[str, None],
        amount: Union[int, None],
        overwrite=False,
        multithreading=True,
        display=False,
    ):
        self.mode = mode
        self.filename = filename
        self.directory = directory
        self.output = output
        self.database = database
        self.amout = amount
        self.overwrite = overwrite
        self.multithreading = multithreading
        self.display = display

    def create(self):
        create_db(self.amout, self.database)

    def detect(self):
        file_paths, output_folder = get_arg_paths(self)

        res = []
        for path in file_paths:
            print()
            print(path)
            # Read Image
            img = cv2.imread(path)

            # Detect Faces
            faces = conditional_call(
                detect_faces_threads, detect_faces, self.multithreading, img
            )

            if not args.output == None:
                output_path = output_folder / (path.stem + "_swapped" + path.suffix)
                cv2.imwrite(output_path, frame_faces(img, faces, True))

            res.append([img, faces])

        if self.display:
            display_imgs([r[0] for r in res], faces=[f[1] for f in res])

        return res

    def blur(self):
        file_paths, output_folder = get_arg_paths(self)

        res = []
        for path in file_paths:
            print()
            print(path)
            # Read Image
            img = cv2.imread(path)

            # Detect Faces
            faces = conditional_call(
                detect_faces_threads, detect_faces, self.multithreading, img
            )

            # Blur Faces
            output_img = blur_faces(img, faces)

            if not args.output == None:
                output_path = output_folder / (path.stem + "_swapped" + path.suffix)
                cv2.imwrite(output_path, output_img)

            res.append([output_img, faces])

        if self.display:
            display_imgs([r[0] for r in res], faces=[f[1] for f in res])

        return res

    def replace(self):
        file_paths, output_folder = get_arg_paths(self)

        db = read_db(self.database)

        res = []
        for path in file_paths:
            print()
            print(path)
            # Read Image
            img = cv2.imread(path)

            # Detect Faces
            faces = conditional_call(
                detect_faces_threads, detect_faces, self.multithreading, img
            )

            # Replace Faces
            output_img = conditional_call(
                tinyface.swap_faces_db_threads,
                tinyface.swap_faces_db,
                self.multithreading,
                img,
                faces,
                db,
            )

            if not args.output == None:
                output_path = output_folder / (path.stem + "_swapped" + path.suffix)
                cv2.imwrite(output_path, output_img)

            res.append([output_img, faces])

        if self.display:
            display_imgs([r[0] for r in res], faces=[f[1] for f in res])

        return res

    def execute(self):
        if self.mode == None or self.mode == "replace":
            self.replace()
        elif self.mode == "create":
            self.create()
        elif self.mode == "blur":
            self.blur()
        elif self.mode == "detect":
            self.detect()


if __name__ == "__main__":
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
    parser.add_argument("-di", "--display", action="store_true", help="Display results")

    ## Parse Arguments
    args = parser.parse_args()

    replacer = FaceReplace(
        mode=args.mode,
        filename=args.filename,
        directory=args.directory,
        output=args.output,
        database=args.database,
        amount=args.amount,
        overwrite=args.overwrite,
        multithreading=(not args.no_multithreading),
        display=args.display,
    )

    replacer.execute()
