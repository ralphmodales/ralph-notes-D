import cv2
import mediapipe as mp
import numpy as np

# be sure to install the dependencies first
# pip install opencv-python mediapipe numpy

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # black canvas
    canvas = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # process image with mediapipe 
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)
    
    # for face 
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # face outline
            face_outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
            for i in range(len(face_outline)):
                pt1 = face_landmarks.landmark[face_outline[i]]
                pt2 = face_landmarks.landmark[face_outline[(i + 1) % len(face_outline)]]
                x1, y1 = int(pt1.x * canvas.shape[1]), int(pt1.y * canvas.shape[0])
                x2, y2 = int(pt2.x * canvas.shape[1]), int(pt2.y * canvas.shape[0])
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # for hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                canvas,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

    # display result
    cv2.imshow('mwhahahaha', canvas)
    print("SOYBOYYYYYYYYY")

    # close when esc was pressed
    if cv2.waitKey(5) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
