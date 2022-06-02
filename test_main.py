import mediapipe as mp
import numpy as np
import cv2 as cv
from utils.CalculateAngle import get_shoulder_L
mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils
pose=mp_pose.Pose(static_image_mode=False,enable_segmentation=True,min_detection_confidence=0.5)
capture=cv.VideoCapture(0)
cv.namedWindow("img",0)
while capture.isOpened():
    ret,frame=capture.read()
    if not ret:
        break
    rgb_image=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results=pose.process(rgb_image)
    print("shoulder_L:",get_shoulder_L(results.pose_landmarks.landmark))
    
    mp_drawing.draw_landmarks(frame, results.pose_landmarks)
    cv.imshow("img", frame)
    key=cv.waitKey(1)
    if key==27:
        cv.destroyAllWindows()
        capture.release()
        break