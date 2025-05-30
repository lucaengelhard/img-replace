from tinyface import TinyFace, FacePair, VisionFrame, Face

import random


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
        temp_vision_frame = vision_frame.copy()

        for face in faces:
            possible_dest = random.choice(options)

            # while not self._is_similar_face(face, possible_dest):
            #     print("not similar enough")
            #     possible_dest = random.choice(options)

            temp_vision_frame = self.swapper.swap_face(
                temp_vision_frame, possible_dest, face
            )

            temp_vision_frame = self.enhancer.enhance_face(temp_vision_frame, face)

        return temp_vision_frame
