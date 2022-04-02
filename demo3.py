import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_tracking_confidence = 0.5, min_detection_confidence = 0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignore camera frame")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for landmark in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_style.get_default_hand_landmarks_style(),
                    mp_drawing_style.get_default_hand_connections_style())
        
        cv2.imshow('Hand Estimator', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()



