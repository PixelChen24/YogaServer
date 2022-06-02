import mediapipe as mp
import numpy as np
Radian=np.pi/180


def get_shoulder_L(pose_landmarks):
    pin1=pose_landmarks[23]
    pin2=pose_landmarks[13]
    joint=pose_landmarks[11]
    return calAngle(pin1, pin2, joint)
def get_shoulder_R(pose_landmarks):
    pin1=pose_landmarks[24]
    pin2=pose_landmarks[14]
    joint=pose_landmarks[12]
    return calAngle(pin1, pin2, joint)


def get_elbow_L(pose_landmarks):
    pin1=pose_landmarks[11]
    pin2=pose_landmarks[15]
    joint=pose_landmarks[13]
    return calAngle(pin1, pin2, joint)
def get_elbow_R(pose_landmarks):
    pin1=pose_landmarks[12]
    pin2=pose_landmarks[16]
    joint=pose_landmarks[14]
    return calAngle(pin1, pin2, joint)


def get_hip_L(pose_landmarks):
    pin1=pose_landmarks[11]
    pin2=pose_landmarks[25]
    joint=pose_landmarks[23]
    return calAngle(pin1, pin2, joint)
def get_hip_R(pose_landmarks):
    pin1=pose_landmarks[12]
    pin2=pose_landmarks[26]
    joint=pose_landmarks[24]
    return calAngle(pin1, pin2, joint)


def get_knee_L(pose_landmarks):
    pin1=pose_landmarks[23]
    pin2=pose_landmarks[27]
    joint=pose_landmarks[25]
    return calAngle(pin1, pin2, joint)
def get_knee_R(pose_landmarks):
    pin1=pose_landmarks[24]
    pin2=pose_landmarks[28]
    joint=pose_landmarks[26]
    return calAngle(pin1, pin2, joint)



def calAngle(pin1,pin2,joint):
    """
    计算由pin1,joint,pin2构成的角的夹角
    """
    vector1=np.array([[pin1.x-joint.x,pin1.y-joint.y]])
    vector2=np.array([[pin2.x-joint.x,pin2.y-joint.y]])
    inner_dot=np.dot(vector1,vector2.T)
    length1=np.linalg.norm(vector1)
    length2=np.linalg.norm(vector2)
    return np.arccos(1.0*inner_dot/length1/length2)/Radian