import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(min_detection_confidence = 0.5) as face:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignore camera frame")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.detections:
            for detection in result.detections:
                mp_drawing.draw_detection(image, detection)

        cv2.imshow('Face Detector', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()

