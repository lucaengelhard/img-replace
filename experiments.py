import cv2
import matplotlib.pyplot as plt

from tinyface import FacePair, TinyFace

base_img = "./imgs/base_img.jpg"
face1 = "./imgs/face1.png"
face2 = "./imgs/face2.png"
face3 = "./imgs/face3.png"
face4 = "./imgs/face4.png"
luca = "./imgs/luca.jpg"
bus = "./imgs/bus.jpg"

input_img = cv2.imread(bus)
reference_img = cv2.imread(luca)
destination_img = cv2.imread(face3)

tinyface = TinyFace()

tinyface.prepare()

faces = tinyface.get_many_faces(input_img)
reference_face = tinyface.get_one_face(reference_img)
destination_face = tinyface.get_one_face(destination_img)

# print("+++++++++++++++++++++")
# print(len(faces))

for face in faces:
    # print(face.bounding_box)
    x = int(face.bounding_box[0])
    y = int(face.bounding_box[1])
    w = int(face.bounding_box[2])
    h = int(face.bounding_box[3])
    cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 4)

shifted_faces = [faces[-1]] + faces[:-1]

# output_img = tinyface.swap_faces(
#     input_img,
#     face_pairs=[
#         FacePair(reference=faces[i], destination=shifted_faces[i])
#         for i in range(len(faces))
#     ],
# )

# output_img = tinyface.swap_faces(
#     input_img,
#     face_pairs=[FacePair(reference=reference_face, destination=destination_face)],
# )

# FacePair(reference=reference_face, destination=destination_face)

# cv2.imwrite("out.jpg", output_img)

plt.figure(figsize=(20, 10))
plt.imshow(input_img)
plt.axis("off")
plt.show()


# output_img = tinyface.swap_face(input_img, reference_face, destination_face)

# output_img = tinyface.swap_faces(
#     input_img,
#     face_pairs=[FacePair(reference=reference_face, destination=destination_face)],
# )
# cv2.imwrite("out.jpg", output_img)


# import cv2
# import cv2.data
# import matplotlib.pyplot as plt

# import face_recognition

# imagePath = "./imgs/51821190129_f60a12a754_h.jpg"

# img = cv2.imread(imagePath)

# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face_classifier = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# face = face_classifier.detectMultiScale(
#     gray_image, scaleFactor=1.05, minNeighbors=2, minSize=(1, 1)
# )


# for x, y, w, h in face:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)


# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(20, 10))
# plt.imshow(img_rgb)
# plt.axis("off")
# plt.show()

# image = face_recognition.load_image_file(imagePath)

# import matplotlib.pyplot as plt

# img = cv2.imread(base_img)

# height, width, _ = img.shape


# if faces is not None:
#     for face in faces:
#         # parameters: x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm

#         # bouding box
#         box = list(map(int, face[:4]))
#         color = (0, 255, 0)
#         cv2.rectangle(img, box, color, 5)

#         # confidence
#         confidence = face[-1]
#         confidence = "{:.2f}".format(confidence)
#         position = (box[0], box[1] - 10)
#         cv2.putText(
#             img,
#             confidence,
#             position,
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             color,
#             3,
#             cv2.LINE_AA,
#         )

# plt.figure()
# plt.imshow(img)
# plt.axis("off")
# plt.show()
