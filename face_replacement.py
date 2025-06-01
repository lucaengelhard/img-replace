from tinyface import TinyFace, FacePair, VisionFrame, Face
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        print(" - Swapping faces (Single Thread)")
        temp_vision_frame = vision_frame.copy()

        for face in tqdm(faces):
            # possible_dest = random.choice(options)

            temp_vision_frame = self.swapper.swap_face(
                temp_vision_frame, get_closest_face(face, options)[1], face
            )

            temp_vision_frame = self.enhancer.enhance_face(temp_vision_frame, face)

        return temp_vision_frame

    def swap_faces_db_threads(
        self, vision_frame: VisionFrame, faces: list[Face], options: list[Face]
    ):
        print(" - Swapping faces")
        temp_vision_frame = vision_frame.copy()

        print(" - Creating face pairs")
        pairs = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(get_closest_face, face, options) for face in faces
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                pairs.append(future.result())

        print()
        print(" - Applying face pairs")
        for pair in tqdm(pairs):
            temp_vision_frame = self.swapper.swap_face(
                temp_vision_frame, pair[1], pair[0]
            )

            temp_vision_frame = self.enhancer.enhance_face(temp_vision_frame, pair[0])

        return temp_vision_frame


def get_closest_face(face: Face, db: list[Face]):
    target = face.normed_embedding

    options = np.vstack([f.normed_embedding for f in db])

    similarities = options @ target

    best = np.argmax(similarities)

    return face, db[best]
