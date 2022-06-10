import sys
from utils.DataBase import check_database
from utils.CalculateAngle import get_elbow_L,get_elbow_R,get_hip_L,get_hip_R,get_knee_L,get_knee_R,get_shoulder_L,get_shoulder_R
from utils.Score import get_score
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2 as cv

from flask import Flask,request
import flask
app=Flask(__name__)


mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils
pose_detector=mp_pose.Pose(static_image_mode=False,enable_segmentation=True,min_detection_confidence=0.5)

sports=["gangling"]

@app.route("/JudgeScore",methods=["POST"])
def JudgeScoreResponse():
    if flask.request.files.get("image"):
        image_bytes=flask.request.files["image"].read()
        image_numpy=np.asarray(bytearray(image_bytes),dtype=np.uint8)
        image_opencv=cv.imdecode(image_numpy,-1)
        image_rgb=cv.cvtColor(image_opencv, cv.COLOR_BGR2RGB)
        sports_type=int(flask.request.files["type"].read())
    return JudgeScore(sports_type=sports_type, image=image_rgb)

def JudgeScore(sports_type,image):
    return_dict={
        "status":"Fail",
        "score":[0,0,0],
        "suggestions":["No suggestions","No suggestions","No suggestions"],
        "points":[
            [1,2]
        ]
    }

    sports_name=sports[sports_type]
    std_csv_file_path="StdPoseDatabase/Points/"+sports_name+".csv"
    std_data=np.array(pd.read_csv(std_csv_file_path))
    std_angles=std_data[:-1,:][0]
    std_weights=std_data[-1,:]
    print(std_angles,std_weights)


    if sports_type is None or image is None:
        return_dict["status"]="No Params"
        return return_dict
    detect_result=pose_detector.process(image)
    if detect_result.pose_landmarks is None:
        return_dict["status"]="No detection results"
        return return_dict

    pose_landmark=detect_result.pose_landmarks.landmark  # 获取33个检测结果 
    points=[]
    for i in range(33):
        point=pose_landmark[i]
        points.append([point.x,point.y])
    return_dict["points"]=points  # 将检测点返回

    # 计算各个关键角度
    shoulder_L=get_shoulder_L(pose_landmark)
    shoulder_R=get_shoulder_R(pose_landmark)
    elbow_L=get_elbow_L(pose_landmark)
    elbow_R=get_elbow_R(pose_landmark)
    hip_L=get_hip_L(pose_landmark)
    hip_R=get_hip_R(pose_landmark)
    knee_L=get_knee_L(pose_landmark)
    knee_R=get_knee_R(pose_landmark)
    score=get_score(standard_angle=std_angles,measured_angle=np.array([shoulder_L,shoulder_R,elbow_L,elbow_R,hip_L,hip_R,knee_L,knee_R]),weights=std_weights)

    return_dict["score"][0]=score
    return_dict["status"]="Success"
    
    return return_dict
    

if __name__=="__main__":
    app.run(port="5000")
    
    # video_capture=cv.VideoCapture(0)
    # cv.namedWindow("img",0)
    # while video_capture.isOpened():
    #     ret,frame=video_capture.read()
    #     if not ret:
    #         break
    #     rgb_frame=cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    #     detect_result=pose_detector.process(rgb_frame)
    #     if detect_result.pose_landmarks is None:
    #         continue
    #     pose_landmark=list(detect_result.pose_landmarks.landmark)
        
    #     shoulder_L=get_shoulder_L(pose_landmark)
    #     shoulder_R=get_shoulder_R(pose_landmark)
    #     elbow_L=get_elbow_L(pose_landmark)
    #     elbow_R=get_elbow_R(pose_landmark)
    #     score=get_score(standard_angle=std,measured_angle=np.array([shoulder_L,shoulder_R,elbow_L,elbow_R]),weights=w)

    #     mp_drawing.draw_landmarks(image=frame, 
    #         landmark_list=detect_result.pose_landmarks,
    #         connections=mp_pose.POSE_CONNECTIONS
    #     )
    #     frame=cv.flip(frame, 1)  # 水平翻转画面
    #     cv.imshow("img", frame)
    #     print("Your Score is:%.2f\n"%(score*100))
    #     key=cv.waitKey(1)
    #     if key==27:
    #         cv.destroyAllWindows()
    #         video_capture.release()
    #         break
