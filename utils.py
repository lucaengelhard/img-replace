from tinyface import Face
from typing import Union
import numpy as np
import cv2
import numpy
from pathlib import Path
import defaults


def frame_faces(img: np.ndarray, faces: list[Face], features=False, scale=1):
    """

    :param img: The image to compute
    :param faces: A list of Faces
    :param feature: Decides if features (eyes, nose, mouth) should also be marked
    :param scale: scales the thickness of the drawn lines
    :return: a new image
    """

    newImg = img.copy()
    for face in faces:
        x = int(face.bounding_box[0])
        y = int(face.bounding_box[1])
        w = int(face.bounding_box[2])
        h = int(face.bounding_box[3])
        cv2.rectangle(
            newImg,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2 * scale,
        )

        if features:
            for landmark in face.landmark_5:
                cv2.circle(
                    newImg,
                    (numpy.uint(landmark[0]), numpy.uint(landmark[1])),
                    2 * scale,
                    (255, 0, 0),
                    1 * scale,
                ),
    return newImg


def scale_image_to(img, max_w, max_h):
    """
    Scales an image to adhere to a given width and height

    :param img: The image to compute
    :param max_w: the maximum width of the resulting image
    :param max_h: the maximum height of the resulting image
    :return: a new image
    """

    # Get original dimensions
    height, width = img.shape[:2]

    #   Calculate the scaling factor
    scale_w = max_w / width
    scale_h = max_h / height
    scale = min(scale_w, scale_h)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def get_image_paths(directory):
    """
    Get the path of every image within a directory

    :return: a list of path strings
    """

    directory_path = Path(directory)
    return [
        str(file)
        for file in directory_path.rglob("*")
        if file.suffix.lower() in defaults.IMAGE_EXTENSIONS
    ]


def parse_path(str):
    """ """

    return Path(str).expanduser().resolve()


def get_arg_paths(args):
    """
    Get the file paths and the output folder from the cli args

    """

    file_paths = [parse_path(defaults.TEST_PATH)]
    output_folder = None

    if not args.filename == None or not args.directory == None:
        file_paths.clear()
        if not args.filename == None:
            file_paths.append(parse_path(args.filename))

        if not args.directory == None:
            file_paths.extend(get_image_paths(parse_path(args.directory)))

    if not args.output == None:
        output_folder = parse_path(args.output)
        output_folder.mkdir(parents=True, exist_ok=True)

    return file_paths, output_folder


def display_img(img, scale=(1920, 1080), faces=None):
    if faces != None:
        cv2.imshow(
            "img-replace", scale_image_to(frame_faces(img, faces), scale[0], scale[1])
        )
    else:
        cv2.imshow("img-replace", scale_image_to(img, scale[0], scale[1]))

    return cv2.waitKey(0)


def display_imgs(imgs: list, scale=(1920, 1080), faces=None):
    print()
    print(f" - Displaying images ({len(imgs)})")

    for i, img in enumerate(imgs):
        i_faces = None
        if faces != None:
            i_faces = faces[i]

        key = display_img(img, scale, i_faces)

        if key == 27:
            break

    cv2.destroyAllWindows()


def s_print(
    silent=False,
    *args,
    **kwargs,
):
    """
    Wrapper for the print function, that doesn't print if silent == True

    """
    if not silent:
        print(*args, **kwargs)


def get_img(img_input: Union[str, np.ndarray]) -> np.ndarray:
    """ """

    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise ValueError(f"Could not read image from path: {img_input}")
    elif isinstance(img_input, np.ndarray):
        img = img_input

    return img


def conditional_call(func1, func2, condition, *args, **kwargs):
    """
    Calls either func1 or func2 based on the value of condition.

    Parameters:
        func1 (callable): First function to potentially call.
        func2 (callable): Second function to potentially call.
        condition (bool): If True, calls func1; if False, calls func2.
        *args: Positional arguments to pass to the chosen function.
        **kwargs: Keyword arguments to pass to the chosen function.

    Returns:
        The result of the chosen function call.
    """
    if condition:
        return func1(*args, **kwargs)
    else:
        return func2(*args, **kwargs)
