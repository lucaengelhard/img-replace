from tinyface import TinyFace, FacePair, VisionFrame, Face

import random

import numpy as np
from tqdm import tqdm


class Modified_TinyFace(TinyFace):
    # Version 1: Swap Faces mit vordefinierten Paaren die einfach geswapped werden
    def swap_faces(self, vision_frame: VisionFrame, face_pairs: list[FacePair]):
        temp_vision_frame = vision_frame.copy()

        for pair in face_pairs:
            reference_face = pair.reference
            destination_face = pair.destination

            temp_vision_frame = self.swapper.swap_face(
                temp_vision_frame, destination_face, reference_face
            )

            temp_vision_frame = self.enhancer.enhance_face(
                temp_vision_frame, reference_face
            )
        return temp_vision_frame

    def swap_faces_db(
        self, vision_frame: VisionFrame, faces: list[Face], options: list[Face]
    ):
        print(" - Swapping faces")
        temp_vision_frame = vision_frame.copy()

        for face in tqdm(faces):
            # possible_dest = random.choice(options)

            temp_vision_frame = self.swapper.swap_face(
                temp_vision_frame, get_closest_face(face, options), face
            )

            temp_vision_frame = self.enhancer.enhance_face(temp_vision_frame, face)

        return temp_vision_frame


# TODO: Make mor efficient (Vectorisation)
def get_closest_face(face: Face, db: list[Face]):
    lowest = None
    res = db[0]

    for option in db:
        cos_distance = np.dot(face.normed_embedding, option.normed_embedding)

        if lowest == None or cos_distance < lowest:
            lowest = cos_distance
            res = option

    return res
