import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class YOLOObjectDetector:
    def __init__(self, weights_path='yolov3.weights', 
                 config_path='yolov3.cfg', 
                 classes_path='coco.names'):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.output_layers = [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        outs = self.net.forward(self.output_layers)
        
        boxes, confidences, class_ids = [], [], []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x, center_y = map(int, detection[:2] * [width, height])
                    w, h = map(int, detection[2:4] * [width, height])
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        return boxes, confidences, class_ids, indexes

MESH_ANNOTATIONS = {
    'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
    'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
    'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
    'rightEyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'leftEyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
}

class FaceAnalyzer:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def is_mouth_open(self, face_landmarks):
        upper_lip = [0, 267, 269, 270, 408, 306, 292, 325, 446, 361]
        lower_lip = [17, 84, 314, 405, 321, 375, 291, 409, 270, 269]

        upper_y = [face_landmarks.landmark[idx].y for idx in upper_lip]
        lower_y = [face_landmarks.landmark[idx].y for idx in lower_lip]

        return abs(np.mean(lower_y) - np.mean(upper_y)) > 0.07

    def convert_landmarks_to_points(self, canvas, landmark_indices, face_landmarks):
        return np.array([
            (int(face_landmarks.landmark[idx].x * canvas.shape[1]), 
             int(face_landmarks.landmark[idx].y * canvas.shape[0]))
            for idx in landmark_indices
        ]).reshape((-1, 1, 2))

    def draw_eye_outline(self, canvas, face_landmarks):
        right_eye_points = MESH_ANNOTATIONS['rightEyeUpper0'] + list(reversed(MESH_ANNOTATIONS['rightEyeLower0']))
        left_eye_points = MESH_ANNOTATIONS['leftEyeUpper0'] + list(reversed(MESH_ANNOTATIONS['leftEyeLower0']))
        
        right_points = self.convert_landmarks_to_points(canvas, right_eye_points, face_landmarks)
        left_points = self.convert_landmarks_to_points(canvas, left_eye_points, face_landmarks)
        
        cv2.polylines(canvas, [right_points], True, (255, 0, 0), 2)  
        cv2.polylines(canvas, [left_points], True, (255, 0, 0), 2)  

    def draw_iris_outline(self, canvas, face_landmarks):
        right_iris = [474, 475, 476, 477]
        left_iris = [469, 470, 471, 472]
        
        right_points = self.convert_landmarks_to_points(canvas, right_iris, face_landmarks)
        left_points = self.convert_landmarks_to_points(canvas, left_iris, face_landmarks)
        
        gojo_blue = (230, 180, 50)  
        cv2.polylines(canvas, [right_points], True, gojo_blue, 2)  
        cv2.polylines(canvas, [left_points], True, gojo_blue, 2)

    def draw_eyebrow_outline(self, canvas, face_landmarks):
        right_points = self.convert_landmarks_to_points(canvas, MESH_ANNOTATIONS['rightEyebrow'], face_landmarks)
        left_points = self.convert_landmarks_to_points(canvas, MESH_ANNOTATIONS['leftEyebrow'], face_landmarks)
        
        cv2.polylines(canvas, [right_points], True, (0, 255, 0), 2) 
        cv2.polylines(canvas, [left_points], True, (0, 255, 0), 2)   

    def draw_lip_outline(self, canvas, face_landmarks):
        upper_lip_landmarks = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191,78]
        lower_lip_landmarks = [78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61]
        
        lip_landmarks = upper_lip_landmarks + lower_lip_landmarks
        
        points = self.convert_landmarks_to_points(canvas, lip_landmarks, face_landmarks)

        cv2.polylines(canvas, [points], True, (0, 0, 255), 3)  
        cv2.fillPoly(canvas, [points], (0, 0, 200)) 

class BackgroundManager:
    @staticmethod
    def load_background_image(filepath, target_size):
        try: 
            bg_image = cv2.imread(filepath)
            bg_image = cv2.resize(bg_image, (target_size[1], target_size[0]))
            return bg_image
        except Exception as e:
            print(f"Error loading background image: {e}")
            return None

    @staticmethod
    def select_background(target_size):
        root = tk.Tk()
        root.withdraw()  
        file_path = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            bg = BackgroundManager.load_background_image(file_path, target_size)
            if bg is not None:
                print(f"Background set to: {file_path}")
                return bg
            print("Failed to load background image.")
        else:
            print("No image selected.")
        return None

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    yolo_detector = YOLOObjectDetector()
    face_analyzer = FaceAnalyzer()
    
    current_background = None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        canvas = current_background.copy() if current_background is not None else np.zeros_like(image)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_results = face_analyzer.face_mesh.process(image_rgb)
        hand_results = face_analyzer.hands.process(image_rgb)

        boxes, confidences, class_ids, indexes = yolo_detector.detect_objects(image)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                if face_analyzer.is_mouth_open(face_landmarks):
                    print('SOYBOYYYYY')
               
                face_analyzer.draw_eye_outline(canvas, face_landmarks)
                face_analyzer.draw_iris_outline(canvas, face_landmarks)  
                face_analyzer.draw_eyebrow_outline(canvas, face_landmarks)  
                face_analyzer.draw_lip_outline(canvas, face_landmarks)

                mp_drawing.draw_landmarks(
                    canvas,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,  
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(yolo_detector.classes[class_ids[i]])
                confidence = confidences[i]
                
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(canvas, f'{label}: {confidence:.2f}', (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Soyface Detector', canvas)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:  
            break
        elif key == ord('b'): 
            current_background = None
            print("Background reset to black.")
        elif key == ord('i'):
            current_background = BackgroundManager.select_background(image.shape[:2])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
