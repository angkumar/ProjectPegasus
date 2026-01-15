import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Local model path
MODEL_PATH = "pose_landmarker_heavy.task"

# Configure the landmarker for VIDEO mode
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' to quit")

# Accurate pose connections
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    
    # Shoulders
    (11, 12),
    
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    
    # Torso
    (12, 24), (11, 23), (23, 24),
    
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
]

# Define body part colors for better visualization
def get_connection_color(connection):
    """Return color based on body part"""
    start, end = connection
    
    # Face - cyan
    if max(start, end) <= 10:
        return (255, 255, 0)
    
    # Left arm - green
    elif start in [11, 13, 15, 17, 19, 21] or end in [11, 13, 15, 17, 19, 21]:
        return (0, 255, 0)
    
    # Right arm - blue
    elif start in [12, 14, 16, 18, 20, 22] or end in [12, 14, 16, 18, 20, 22]:
        return (255, 0, 0)
    
    # Left leg - yellow
    elif start in [23, 25, 27, 29, 31] or end in [23, 25, 27, 29, 31]:
        return (0, 255, 255)
    
    # Right leg - magenta
    elif start in [24, 26, 28, 30, 32] or end in [24, 26, 28, 30, 32]:
        return (255, 0, 255)
    
    # Torso - white
    else:
        return (255, 255, 255)

def draw_landmarks_on_image(frame, detection_result):
    """Draw pose landmarks and connections on the frame"""
    if not detection_result.pose_landmarks:
        return frame
    
    h, w, _ = frame.shape
    
    for pose_landmarks in detection_result.pose_landmarks:
        # Draw connections (skeleton lines) with colors
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                start_landmark = pose_landmarks[start_idx]
                end_landmark = pose_landmarks[end_idx]
                
                # Convert normalized coordinates to pixel coordinates
                start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                
                # Get color for this connection
                color = get_connection_color(connection)
                
                # Draw thicker line for better visibility
                cv2.line(frame, start_point, end_point, color, 3)
        
        # Draw landmarks (red dots on joints)
        for idx, landmark in enumerate(pose_landmarks):
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # Different sizes for different landmarks
            if idx in [11, 12, 23, 24]:  # Major joints (shoulders, hips)
                radius = 8
            elif idx <= 10:  # Face landmarks
                radius = 4
            else:  # Other joints
                radius = 6
            
            # Draw red circle for each joint
            cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)
            
            # Draw white border around each dot
            cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
    
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap in MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Timestamp in milliseconds
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    # Detect synchronously
    result = landmarker.detect_for_video(mp_image, timestamp)

    # Draw the skeleton and joints
    frame = draw_landmarks_on_image(frame, result)

    # Show frame with overlay
    cv2.imshow("Full Body Skeleton Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()