import cv2
import mediapipe as mp
import pybullet as p
import pybullet_data
import numpy as np
import math
import time
from ultralytics import YOLO

CAMERA_ID = 0
WINDOW_MAIN = "YOLO Gesture + MediaPipe PyBullet Avatar"
SCALE = 1.2
SMOOTHING = 0.35

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=35, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

def create_capsule(radius, height, color):
    visual_shape = p.createVisualShape(shapeType=p.GEOM_CAPSULE, radius=radius, length=height, rgbaColor=color)
    collision_shape = p.createCollisionShape(shapeType=p.GEOM_CAPSULE, radius=radius, height=height)
    body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape)
    return body_id

def create_sphere(radius, color):
    visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape)
    return body_id

HEAD = create_sphere(0.08, [0.9, 0.7, 0.1, 1])
TORSO = create_capsule(0.09, 0.25, [0.1, 0.5, 0.2, 1])
LEFT_UPPER_ARM = create_capsule(0.035, 0.18, [0.1, 0.2, 0.8, 1])
LEFT_FOREARM = create_capsule(0.03, 0.18, [0.8, 0.6, 0.1, 1])
RIGHT_UPPER_ARM = create_capsule(0.035, 0.18, [0.1, 0.2, 0.8, 1])
RIGHT_FOREARM = create_capsule(0.03, 0.18, [0.8, 0.6, 0.1, 1])
LEFT_THIGH = create_capsule(0.045, 0.22, [0.2, 0.6, 0.2, 1])
LEFT_SHIN = create_capsule(0.035, 0.22, [0.8, 0.7, 0.1, 1])
RIGHT_THIGH = create_capsule(0.045, 0.22, [0.2, 0.6, 0.2, 1])
RIGHT_SHIN = create_capsule(0.035, 0.22, [0.8, 0.7, 0.1, 1])
LEFT_SHOULDER_MARKER = create_sphere(0.045, [0.1, 0.1, 0.8, 1])
RIGHT_SHOULDER_MARKER = create_sphere(0.045, [0.1, 0.1, 0.8, 1])
LEFT_ELBOW_MARKER = create_sphere(0.04, [0.8, 0.1, 0.1, 1])
RIGHT_ELBOW_MARKER = create_sphere(0.04, [0.8, 0.1, 0.1, 1])
LEFT_HIP_MARKER = create_sphere(0.05, [0.2, 0.2, 0.8, 1])
RIGHT_HIP_MARKER = create_sphere(0.05, [0.2, 0.2, 0.8, 1])
LEFT_KNEE_MARKER = create_sphere(0.045, [0.8, 0.1, 0.1, 1])
RIGHT_KNEE_MARKER = create_sphere(0.045, [0.8, 0.1, 0.1, 1])

def set_body_between(body_id, start, end):
    start = np.array(start, dtype=np.float32)
    end = np.array(end, dtype=np.float32)
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 0.001:
        return
    center = (start + end) / 2.0
    direction = direction / length
    default_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(default_axis, direction)
    dot = np.dot(default_axis, direction)
    dot = np.clip(dot, -1.0, 1.0)
    angle = math.acos(dot)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 0.0001:
        quaternion = [0, 0, 0, 1]
    else:
        axis = cross / cross_norm
        quaternion = p.getQuaternionFromAxisAngle(axis.tolist(), angle)
    p.resetBasePositionAndOrientation(body_id, center.tolist(), quaternion)

previous_positions = {}

def smooth_position(name, position):
    position = np.array(position)
    if name not in previous_positions:
        previous_positions[name] = position
    previous_positions[name] = previous_positions[name] * (1.0 - SMOOTHING) + position * SMOOTHING
    return previous_positions[name]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

yolo_model = YOLO('yolov26n.pt') 

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def landmark_to_world(landmark):
    x = (landmark.x - 0.5) * SCALE
    y = -landmark.z * SCALE * 0.7
    z = (1.0 - landmark.y) * SCALE * 0.7
    return np.array([x, y, z])

def capture_pybullet_frame():
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(320, 240, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_img = np.reshape(rgb_img, (height, width, 4))
    rgb_img = rgb_img[:, :, :3]
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    return rgb_img

print("==========================================")
print(" YOLO GESTURE + MEDIAPIPE -> PYBULLET")
print(" Press Q to exit")
print("==========================================")

with mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera not detected")
            break
        
        frame = cv2.flip(frame, 1)
        
        yolo_results = yolo_model(frame, verbose=False)
        
        annotated_frame = yolo_results[0].plot()
        
        gesture_text = "None"
        if len(yolo_results[0].boxes) > 0:
            top_box = yolo_results[0].boxes[0]
            class_id = int(top_box.cls[0])
            gesture_text = yolo_model.names[class_id]
            
        cv2.putText(annotated_frame, f"YOLO Detection: {gesture_text}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmark_to_world(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
            right_shoulder = landmark_to_world(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            left_elbow = landmark_to_world(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
            right_elbow = landmark_to_world(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
            left_wrist = landmark_to_world(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
            right_wrist = landmark_to_world(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
            left_hip = landmark_to_world(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
            right_hip = landmark_to_world(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
            left_knee = landmark_to_world(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])
            right_knee = landmark_to_world(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE])
            left_ankle = landmark_to_world(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
            right_ankle = landmark_to_world(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])
            

            left_shoulder = smooth_position("left_shoulder", left_shoulder)
            right_shoulder = smooth_position("right_shoulder", right_shoulder)
            left_elbow = smooth_position("left_elbow", left_elbow)
            right_elbow = smooth_position("right_elbow", right_elbow)
            left_wrist = smooth_position("left_wrist", left_wrist)
            right_wrist = smooth_position("right_wrist", right_wrist)
            left_hip = smooth_position("left_hip", left_hip)
            right_hip = smooth_position("right_hip", right_hip)
            left_knee = smooth_position("left_knee", left_knee)
            right_knee = smooth_position("right_knee", right_knee)
            left_ankle = smooth_position("left_ankle", left_ankle)
            right_ankle = smooth_position("right_ankle", right_ankle)

            body_center = (left_hip + right_hip) / 2.0
            shoulder_center = (left_shoulder + right_shoulder) / 2.0
            hip_center = (left_hip + right_hip) / 2.0
            
            set_body_between(TORSO, hip_center, shoulder_center)
            
            head_offset = np.array([0, 0, 0.2])
            head_position = shoulder_center + head_offset
            p.resetBasePositionAndOrientation(HEAD, head_position.tolist(), [0, 0, 0, 1])
            
            set_body_between(LEFT_UPPER_ARM, left_shoulder, left_elbow)
            set_body_between(LEFT_FOREARM, left_elbow, left_wrist)
            set_body_between(RIGHT_UPPER_ARM, right_shoulder, right_elbow)
            set_body_between(RIGHT_FOREARM, right_elbow, right_wrist)
            set_body_between(LEFT_THIGH, left_hip, left_knee)
            set_body_between(LEFT_SHIN, left_knee, left_ankle)
            set_body_between(RIGHT_THIGH, right_hip, right_knee)
            set_body_between(RIGHT_SHIN, right_knee, right_ankle)
            
            marker_positions = {
                LEFT_SHOULDER_MARKER: left_shoulder,
                RIGHT_SHOULDER_MARKER: right_shoulder,
                LEFT_ELBOW_MARKER: left_elbow,
                RIGHT_ELBOW_MARKER: right_elbow,
                LEFT_HIP_MARKER: left_hip,
                RIGHT_HIP_MARKER: right_hip,
                LEFT_KNEE_MARKER: left_knee,
                RIGHT_KNEE_MARKER: right_knee
            }
            for marker_id, position in marker_positions.items():
                p.resetBasePositionAndOrientation(marker_id, position.tolist(), [0, 0, 0, 1])
            
            mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3), 
                                     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
        
        pybullet_frame = capture_pybullet_frame()
        pybullet_frame = cv2.resize(pybullet_frame, (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2))
        frame_resized = cv2.resize(annotated_frame, (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2))
        
        combined_frame = np.hstack((frame_resized, pybullet_frame))

        cv2.putText(combined_frame, "Camera + YOLO", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined_frame, "PyBullet Avatar", (frame_resized.shape[1] + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(WINDOW_MAIN, combined_frame)
        p.stepSimulation()
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
p.disconnect()
