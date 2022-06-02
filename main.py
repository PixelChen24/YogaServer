import sys
from utils.DataBase import check_database
from utils.CalculateAngle import get_elbow_L,get_elbow_R,get_hip_L,get_hip_R,get_knee_L,get_knee_R,get_shoulder_L,get_shoulder_R
from utils.Score import get_score
import mediapipe as mp
import numpy as np
import cv2 as cv


std=np.array([90,90,90,90])  # 测试动作为举哑铃动作，要求肩关节和肘关节90度
w=np.array([0.25,0.25,0.25,0.25])
if __name__=="__main__":
    mp_pose=mp.solutions.pose
    mp_drawing=mp.solutions.drawing_utils
    pose_detector=mp_pose.Pose(static_image_mode=False,enable_segmentation=True,min_detection_confidence=0.5)
    
    video_capture=cv.VideoCapture(0)
    cv.namedWindow("img",0)
    while video_capture.isOpened():
        ret,frame=video_capture.read()
        if not ret:
            break
        rgb_frame=cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        detect_result=pose_detector.process(rgb_frame)
        pose_landmark=detect_result.pose_landmarks.landmark
        shoulder_L=get_shoulder_L(pose_landmark)
        shoulder_R=get_shoulder_R(pose_landmark)
        elbow_L=get_elbow_L(pose_landmark)
        elbow_R=get_elbow_R(pose_landmark)
        score=get_score(standard_angle=std,measured_angle=np.array([shoulder_L,shoulder_R,elbow_L,elbow_R]),weights=w)

        mp_drawing.draw_landmarks(image=frame, 
            landmark_list=detect_result.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS
        )
        frame=cv.flip(frame, 1)  # 水平翻转画面
        cv.imshow("img", frame)
        print("Your Score is:%.2f\n"%(score*100))
        key=cv.waitKey(1)
        if key==27:
            cv.destroyAllWindows()
            video_capture.release()
            break
