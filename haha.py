import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_objectron = mp.solutions.objectron

# from tensorflow/tjms-models
MESH_ANNOTATIONS = {
    'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
    'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'rightEyeUpper1': [247, 30, 29, 27, 28, 56, 190],
    'rightEyeLower1': [130, 25, 110, 24, 23, 22, 26, 112, 243],
    'rightEyeUpper2': [113, 225, 224, 223, 222, 221, 189],
    'rightEyeLower2': [226, 31, 228, 229, 230, 231, 232, 233, 244],
    'rightEyeLower3': [143, 111, 117, 118, 119, 120, 121, 128, 245],
    
    'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
    'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
    'leftEyeUpper1': [467, 260, 259, 257, 258, 286, 414],
    'leftEyeLower1': [359, 255, 339, 254, 253, 252, 256, 341, 463],
    'leftEyeUpper2': [342, 445, 444, 443, 442, 441, 413],
    'leftEyeLower2': [446, 261, 448, 449, 450, 451, 452, 453, 464],
    'leftEyeLower3': [372, 340, 346, 347, 348, 349, 350, 357, 465],

    'rightEyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'leftEyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
}


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

objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_name='Cup'  
)

cap = cv2.VideoCapture(0)

def load_background_image(filepath):
    try: 
        bg_image = cv2.imread(filepath)
        bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
        return bg_image
    except Exception as e:
        print(f"Error loading background image: {e}")
        return None

current_background = None

def select_background():
    global current_background
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        title="Select Background Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
    )
    
    if file_path:
        bg = load_background_image(file_path)
        if bg is not None:
            current_background = bg
            print(f"Background set to: {file_path}")
        else:
            print("Failed to load background image.")
    else:
        print("No image selected.")

def reset_to_black_background():
    global current_background
    current_background = None
    print("Background reset to black.")

def is_mouth_open(face_landmarks):
    upper_lip = [0, 267, 269, 270, 408, 306, 292, 325, 446, 361]
    lower_lip = [17, 84, 314, 405, 321, 375, 291, 409, 270, 269]

    upper_y = [face_landmarks.landmark[idx].y for idx in upper_lip]
    lower_y = [face_landmarks.landmark[idx].y for idx in lower_lip]

    mouth_height = abs(np.mean(lower_y) - np.mean(upper_y))

    return mouth_height > 0.07 # adjust this if not working 

def draw_eye_outline(canvas, face_landmarks):
    right_eye_points = (MESH_ANNOTATIONS['rightEyeUpper0'] + 
                        list(reversed(MESH_ANNOTATIONS['rightEyeLower0'])))
    
    left_eye_points = (MESH_ANNOTATIONS['leftEyeUpper0'] + 
                       list(reversed(MESH_ANNOTATIONS['leftEyeLower0'])))
    
    def convert_landmarks_to_points(landmark_indices):
        points = []
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * canvas.shape[1])
            y = int(landmark.y * canvas.shape[0])
            points.append((x, y))
        return points
    
    right_points = convert_landmarks_to_points(right_eye_points)
    left_points = convert_landmarks_to_points(left_eye_points)
    
    right_points = np.array(right_points, np.int32).reshape((-1, 1, 2))
    left_points = np.array(left_points, np.int32).reshape((-1, 1, 2))
    
    cv2.polylines(canvas, [right_points], True, (255, 0, 0), 2)  
    cv2.polylines(canvas, [left_points], True, (255, 0, 0), 2)  

def draw_objectron_objects(canvas, detection_result):
    for detected_object in detection_result.detected_objects:
        mp_drawing.draw_landmarks(
            canvas,
            detected_object.landmarks_2d,
            mp_objectron.BOX_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        box_center = detected_object.landmarks_2d.landmark[0]
        x = int(box_center.x * canvas.shape[1])
        y = int(box_center.y * canvas.shape[0])

        cv2.putText(
            canvas,
            "Cup", 
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

def draw_iris_outline(canvas, face_landmarks): 
    right_iris = [474, 475, 476, 477]
    left_iris = [469, 470, 471, 472]
    
    def convert_landmarks_to_points(landmark_indices):
        points = []
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * canvas.shape[1])
            y = int(landmark.y * canvas.shape[0])
            points.append((x, y))
        return points
    
    right_iris_points = convert_landmarks_to_points(right_iris)
    left_iris_points = convert_landmarks_to_points(left_iris)
    
    right_iris_points = np.array(right_iris_points, np.int32).reshape((-1, 1, 2))
    left_iris_points = np.array(left_iris_points, np.int32).reshape((-1, 1, 2))
    
    gojo_blue = (230, 180, 50)  
     
    cv2.polylines(canvas, [right_iris_points], True, gojo_blue, 2)  
    cv2.polylines(canvas, [left_iris_points], True, gojo_blue, 2)

def draw_eyebrow_outline(canvas, face_landmarks):
    def convert_landmarks_to_points(landmark_indices):
        points = []
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * canvas.shape[1])
            y = int(landmark.y * canvas.shape[0])
            points.append((x, y))
        return points
    
    right_eyebrow_points = convert_landmarks_to_points(MESH_ANNOTATIONS['rightEyebrow'])
    left_eyebrow_points = convert_landmarks_to_points(MESH_ANNOTATIONS['leftEyebrow'])
    
    right_eyebrow_points = np.array(right_eyebrow_points, np.int32).reshape((-1, 1, 2))
    left_eyebrow_points = np.array(left_eyebrow_points, np.int32).reshape((-1, 1, 2))
    
    cv2.polylines(canvas, [right_eyebrow_points], True, (0, 255, 0), 2) 
    cv2.polylines(canvas, [left_eyebrow_points], True, (0, 255, 0), 2)   

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

    cv2.polylines(canvas, [points], True, (0, 0, 255), 3)  
    cv2.fillPoly(canvas, [points], (0, 0, 200)) 

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    if current_background is not None:
        canvas = current_background.copy()
    else:
        canvas = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)
    objectron_results = objectron.process(image_rgb)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # if mouth is open print soyboy
            if is_mouth_open(face_landmarks):
                print('SOYBOYYYYY')
           
            draw_eye_outline(canvas, face_landmarks)
            draw_iris_outline(canvas, face_landmarks)  
            draw_eyebrow_outline(canvas, face_landmarks)  
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

    if objectron_results.detected_objects:
        draw_objectron_objects(canvas, objectron_results)

    cv2.imshow('Soyface Detector', canvas)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:  
        break
    elif key == ord('b'): 
        reset_to_black_background()
    elif key == ord('i'):
        select_background()

cap.release()
cv2.destroyAllWindows()
