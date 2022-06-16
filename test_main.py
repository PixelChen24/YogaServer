import sys
from utils.DataBase import check_database
from utils.CalculateAngle import get_elbow_L,get_elbow_R,get_hip_L,get_hip_R,get_knee_L,get_knee_R,get_shoulder_L,get_shoulder_R
from utils.Score import get_score,get_suggestion
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2 as cv
import json
from flask import Flask,request
import flask
import base64
app=Flask(__name__)
cnt=0

mp_pose=mp.solutions.pose
mp_drawing=mp.solutions.drawing_utils
pose_detector=mp_pose.Pose(static_image_mode=False,enable_segmentation=True,min_detection_confidence=0.5)

sports=["classic","gangling","lashen"]

@app.route("/JudgeScore",methods=["POST"])
def JudgeScoreResponse():
    
    sports_type=0
    image_rgb=None
    if request.get_json():
        print("Receive request")
        json_data=request.get_json()
        image_str=json_data["image"]
        image_64=base64.b64decode(image_str)
        image_arr=np.fromstring(image_64,np.uint8)
        image_opencv=cv.imdecode(image_arr,cv.IMREAD_COLOR)
        print(image_opencv.shape)
        sports_type=json_data["type"]
    return_result=JudgeScore(sports_type=int(sports_type), image=image_opencv)
    print(return_result)
    return return_result

def JudgeScore(sports_type,image):
    global cnt
    return_dict={
        "status":"Fail",
        "score":0,
        "suggestions":"No suggestions",
        "points":[
            [1,2]
        ]
    }

    sports_name=sports[sports_type]
    std_csv_file_path="StdPoseDatabase/Points/"+sports_name+".csv"
    std_data=np.array(pd.read_csv(std_csv_file_path))
    std_angles=std_data[:-1,:][0]
    std_weights=std_data[-1,:]


    if sports_type is None or image is None:
        return_dict["status"]="No Params"
        return json.dumps(return_dict)
    detect_result=pose_detector.process(image)
    if detect_result.pose_landmarks is None:
        return_dict["status"]="No detection results"
        return json.dumps(return_dict)

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
    measured_angles=np.array([shoulder_L,shoulder_R,elbow_L,elbow_R,hip_L,hip_R,knee_L,knee_R])
    score=get_score(standard_angle=std_angles,measured_angle=measured_angles,weights=std_weights)
    # mp_drawing.draw_landmarks(image=image, 
    #         landmark_list=detect_result.pose_landmarks,
    #         connections=mp_pose.POSE_CONNECTIONS
    #     )
    # cv.imwrite(str(cnt)+"a.jpg", image)
    # cnt+=1
    suggestion=get_suggestion(standard_angle=std_angles, measured_angle=measured_angles, weights=std_weights)
    return_dict["score"]=score
    return_dict["suggestions"]=suggestion
    return_dict["status"]="Success"
    
    return json.dumps(return_dict)

    

if __name__=="__main__":
    # app.config["JSON_AS_ASCII"]=False
    app.run(host="0.0.0.0",port="5000")
    
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
