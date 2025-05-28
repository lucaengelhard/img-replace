import cv2


detector = cv2.FaceDetectorYN.create(
    "./data/face_detection_yunet_2023mar.onnx", "", (0, 0), score_threshold=0.5
)


def recognize_faces(path: str):
    img = cv2.imread(path)
    height, width, _ = img.shape
    detector.setInputSize((width, height))
    _, faces = detector.detect(img)
    res = []

    for face in faces:
        res.append({"x": face[0], "y": face[1], "w": face[2], "h": face[3]})

    return res
