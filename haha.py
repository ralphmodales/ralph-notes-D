import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

def is_mouth_open(face_landmarks):
    upper_lip = [0, 267, 269, 270, 408, 306, 292, 325, 446, 361]
    lower_lip = [17, 84, 314, 405, 321, 375, 291, 409, 270, 269]

    upper_y = [face_landmarks.landmark[idx].y for idx in upper_lip]
    lower_y = [face_landmarks.landmark[idx].y for idx in lower_lip]

    mouth_height = abs(np.mean(lower_y) - np.mean(upper_y))

    return mouth_height > 0.07 # adjust this if not working 

def draw_lip_outline(canvas, face_landmarks):
    upper_lip_landmarks = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191,78]
    lower_lip_landmarks = [78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61]
    
    lip_landmarks = upper_lip_landmarks + lower_lip_landmarks
    
    points = []
    for idx in lip_landmarks:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * canvas.shape[1])
        y = int(landmark.y * canvas.shape[0])
        points.append((x, y))
    
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(canvas, [points], True, (0, 0, 255), 2)  

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # black canvas, you can change it by adding canvas with its rgb
    canvas = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # if mouth is open print soyboy
            if is_mouth_open(face_landmarks):
                print('SOYBOYYYYY')

            draw_lip_outline(canvas, face_landmarks)

            # for face landmarks
            mp_drawing.draw_landmarks(
                canvas,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,  # tesselation for detailed wireframe
                landmark_drawing_spec=None,  
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # for hand landmarks
            mp_drawing.draw_landmarks(
                canvas,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )

    cv2.imshow('Soyface Detector', canvas)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
