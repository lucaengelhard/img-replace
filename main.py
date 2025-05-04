# import cv2
# import cv2.data
# import matplotlib.pyplot as plt

import face_recognition

imagePath = "./imgs/51821190129_f60a12a754_h.jpg"

# img = cv2.imread(imagePath)

# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face_classifier = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# face = face_classifier.detectMultiScale(
#     gray_image, scaleFactor=1.2, minNeighbors=5, minSize=(1, 1)
# )


# for x, y, w, h in face:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)


# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(20, 10))
# plt.imshow(img_rgb)
# plt.axis("off")
# plt.show()

image = face_recognition.load_image_file(imagePath)
