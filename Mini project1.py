import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import time
import os
import pygame  # Import pygame for sound

# Initialize pygame for sound
#pygame.mixer.init()
#sound_file = r'C:\Users\karma\.jupyter\Demo\BPDI\sound'  # Update with your sound file path

pygame.mixer.init()

sound_file = r'C:\Users\karma\.jupyter\Demo\BPDI\sound\beep-06.wav'  # Correct your file path here

try:
    pygame.mixer.music.load(sound_file)
    print("Sound loaded successfully!")
except pygame.error as e:
    print(f"Error loading sound: {e}")
    
# Initialize Mediapipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Counter to keep track of screenshots
screenshot_counter = 0

def calculate_angle(point1, point2, point3):
    # Convert points to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    # Calculate the vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate the angle between the vectors
    angle = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
    return np.degrees(angle)

def is_aligned(shoulder_offset):
    # Threshold for shoulder alignment
    alignment_threshold = 20  # degrees
    return shoulder_offset < alignment_threshold

def is_bad_posture(shoulder_offset, neck_angle, torso_angle):
    # Define thresholds for bad posture
    shoulder_threshold = 20  # degrees
    neck_threshold = 15  # degrees
    torso_threshold = 10  # degrees
    
    if shoulder_offset > shoulder_threshold or neck_angle > neck_threshold or torso_angle > torso_threshold:
        return True
    return False

def send_alert():
    # Simulate sending an alert
    print("ALERT: Bad posture detected!")

# Define the function to save a screenshot
def save_screenshot(frame, counter):
    # Specify the directory where you want to save the screenshots
    save_directory = r'C:\Users\karma\.jupyter\Demo\BPDI\image'  # Use raw string or forward slashes
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Create the full file path and save the frame as an image
    file_name = f'{save_directory}/screenshot_{counter}.png'
    cv2.imwrite(file_name, frame)
    print(f"Good posture detected, screenshot saved as {file_name}")

# Function to play sound when good posture is detected
def play_sound():
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get pose landmarks
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks for shoulder points, neck, and hips (torso)
        landmarks = results.pose_landmarks.landmark

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

        # Check shoulder alignment (offset between left and right shoulders)
        shoulder_offset = np.abs(left_shoulder[1] - right_shoulder[1]) * 100  # in percentage

        # Calculate neck inclination (angle between nose and shoulder line)
        neck_angle = calculate_angle(nose, left_shoulder, right_shoulder)

        # Calculate torso inclination (angle between shoulders and hips)
        torso_angle = calculate_angle(left_shoulder, left_hip, right_hip)

        # Display offset, neck angle, and torso angle
        cv2.putText(frame, f"Shoulder Offset: {shoulder_offset:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Neck Angle: {neck_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Torso Angle: {torso_angle:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Check shoulder alignment
        if not is_aligned(shoulder_offset):
            cv2.putText(frame, "Warning: Misalignment Detected!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Check if bad posture is detected
        if is_bad_posture(shoulder_offset, neck_angle, torso_angle):
            cv2.putText(frame, "Bad Posture!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            send_alert()  # Trigger alert
        else:
            cv2.putText(frame, "Good Posture", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Capture screenshot if good posture is detected
            screenshot_counter += 1
            save_screenshot(frame, screenshot_counter)

            # Play sound for good posture
            play_sound()

    # Display the frame
    cv2.imshow('Posture Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
