import cv2
import numpy as np

from extensions import ImageFaceExtractor, VideoRealFakeDetector


video_cap = cv2.VideoCapture('./data/liveness_flwses_9q1No64xWvMVVe.webm')
flg, frame = video_cap.read()
org_faces, faces = [], []
cnt = 0

while flg:
    if cnt % 2 == 0:
        face = ImageFaceExtractor()(frame)
        face = cv2.resize(face, (224, 224))
        if face is not None:
            faces.append(VideoRealFakeDetector.preprocess_faces(face))
            org_faces.append(face)
    flg, frame = video_cap.read()
    cnt += 1


faces = np.array(faces)
VideoRealFakeDetector().real_fake_detector.setInput(faces)
result = VideoRealFakeDetector().real_fake_detector.forward() + 2
result = 1 / (1 + np.exp(0 - result))
for face, score in zip(org_faces, result[:, 0]):
    cv2.imwrite(f'data/{score:.5f}.jpg', face)
