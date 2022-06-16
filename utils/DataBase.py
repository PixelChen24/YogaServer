import pandas as pd
import numpy as np
import mediapipe as mp
import os
import mediapipe as mp
import cv2 as cv
from CalculateAngle import *

mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils
pose_detector=mp_pose.Pose(static_image_mode=True,enable_segmentation=True,min_detection_confidence=0.5)

def check_database():
    """ 
    系统自检数据库状态
    """
    pass  #TODO


def update_database():
    """
    对于新加入的标准图片产生对应的数据;对于已删除的图片删除对应的数据
    """
    with open("StdPoseDatabase/Management.txt","r") as f:
        recording_list=f.readlines()
    current_image_list=os.listdir("StdPoseDatabase/Images")
    pass #TODO
    
def generate_data(images_dir,save_csv_path="/media/chen/Study and Work/_XDUDocument/___XDU3/软件工程/软件工程项目/YogaServer/StdPoseDatabase/Points",filesuffix=["jpg","png","jpeg"],enable_debug=True):
    img_list=os.listdir(images_dir)
    for img_name in img_list:
        suffix=img_name.split(".")[1]
        sports_name=img_name.split(".")[0]
        if suffix not in filesuffix:
            continue
        img_name=images_dir+"/"+img_name
        img=cv.imread(img_name)
        img_rgb=cv.cvtColor(img, cv.COLOR_BGR2RGB)
        detect_result=pose_detector.process(img_rgb)
        if detect_result.pose_landmarks is None:
            print("Warning: %s does not include body"%img_name)
            continue
        pose_landmark=detect_result.pose_landmarks.landmark
        shoulder_L=get_shoulder_L(pose_landmark)
        shoulder_R=get_shoulder_R(pose_landmark)
        elbow_L=get_elbow_L(pose_landmark)
        elbow_R=get_elbow_R(pose_landmark)
        hip_L=get_hip_L(pose_landmark)
        hip_R=get_hip_R(pose_landmark)
        knee_L=get_knee_L(pose_landmark)
        knee_R=get_knee_R(pose_landmark)
        if enable_debug:
            mp_drawing.draw_landmarks(image=img, 
                landmark_list=detect_result.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS
            )
            cv.imwrite(save_csv_path+"/"+sports_name+".jpg", img)
            print("Save debug pic %s successfully"%img_name)
        to_save=np.array([
            [shoulder_L,shoulder_R,elbow_L,elbow_R,hip_L,hip_R,knee_L,knee_R],
            [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]
        ])
        pd.DataFrame(to_save).to_csv(save_csv_path+"/"+sports_name+".csv",index=None,header=['sl','sr','el','er','hl','hr','kl','kr'])

if __name__=="__main__":
    generate_data("/media/chen/Study and Work/_XDUDocument/___XDU3/软件工程/软件工程项目/YogaServer/StdPoseDatabase/Images")

