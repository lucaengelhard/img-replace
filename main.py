import cv2
import argparse
from typing import Union
from tinyface import VisionFrame, Face

from src.core.face_detection import detect_faces, detect_faces_threads
from src.core.face_replacement import Modified_TinyFace
from src.core.face_creation import read_db, create_db
from src.core.face_blur import blur_faces
from src.utils import display_imgs, frame_faces, get_arg_paths, conditional_call
import src.defaults as defaults

# Setup
tinyface = Modified_TinyFace()
tinyface.prepare()


"""
TODO:
    - Testing
        - Speed
        - Different OS
        - Different PCs
        - Quality of replacements
        - Different Imgs
    - GIMP Plugin
    - Name
        - Maybe reference a famous political photographer?
"""


class FaceReplace:
    def __init__(
        self,
        mode: str = "replace",
        filename: Union[str, None] = None,
        directory: Union[str, None] = None,
        output: Union[str, None] = None,
        database: Union[str, None] = None,
        amount: Union[int, None] = None,
        # overwrite=False,
        multithreading=True,
        display=False,
        cli=False,
    ):
        self.mode = self._check_mode(mode)
        self.filename = filename
        self.directory = directory
        self.output = output
        self.database = database
        self.amout = self._check_amout(amount)
        # self.overwrite = overwrite
        self.multithreading = multithreading
        self.display = display
        self.cli = cli

    def create(self):
        return create_db(self.amout, self.database)

    def detect(self):
        file_paths, output_folder = get_arg_paths(self)

        res: list[tuple[VisionFrame, list[Face]]] = []
        for path in file_paths:
            print()
            print(path)
            # Read Image
            img = cv2.imread(path)

            # Detect Faces
            faces = conditional_call(
                detect_faces_threads, detect_faces, self.multithreading, img
            )

            if not self.output == None:
                output_path = output_folder / (path.stem + "_swapped" + path.suffix)
                cv2.imwrite(output_path, frame_faces(img, faces, True))

            res.append((img, faces))

        if self.display:
            display_imgs([r[0] for r in res], faces=[f[1] for f in res])

        return res

    def blur(self):
        file_paths, output_folder = get_arg_paths(self)

        res: list[tuple[VisionFrame, list[Face]]] = []
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

            if not self.output == None:
                output_path = output_folder / (path.stem + "_swapped" + path.suffix)
                cv2.imwrite(output_path, output_img)

            res.append((output_img, faces))

        if self.display:
            display_imgs([r[0] for r in res], faces=[f[1] for f in res])

        return res

    def replace(self):
        file_paths, output_folder = get_arg_paths(self)

        db, db_name = read_db(self.database, self.cli)

        self.database = db_name

        if db == None:
            print(" - No database found and no new database created. Exiting.")
            return

        res: list[tuple[VisionFrame, list[Face]]] = []
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

            if not self.output == None:
                output_path = output_folder / (path.stem + "_swapped" + path.suffix)
                cv2.imwrite(output_path, output_img)

            res.append((output_img, faces))

        if self.display:
            display_imgs([r[0] for r in res], faces=[f[1] for f in res])

        return res

    def execute(self):
        if self.mode == None or self.mode == "replace":
            return self.mode, self.replace()
        elif self.mode == "create":
            return self.mode, self.create()
        elif self.mode == "blur":
            return self.mode, self.blur()
        elif self.mode == "detect":
            return self.mode, self.detect()

    def _check_mode(self, input: str):
        if input.lower() not in defaults.MODES:
            raise ValueError(f"Mode: {input} is invalid")

        return input

    def _check_amout(self, input: int):
        if input != None and input < 1:
            raise ValueError(f"Amount {input} has to be at least 1")

        return input


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
    )
    parser.add_argument(
        "-a", "--amount", type=int, help="Amount of created faces", default=10
    )
    # parser.add_argument("-ov", "--overwrite", action="store_true", help="Overwrite?")
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
        # overwrite=args.overwrite,
        multithreading=(not args.no_multithreading),
        display=args.display,
        cli=True,
    )

    replacer.execute()
