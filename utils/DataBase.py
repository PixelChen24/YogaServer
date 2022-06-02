import pandas as pd
import numpy as np
import mediapipe as mp
import os
def check_database():
    pass  #TODO
def update_database():
    """
    对于新加入的标准图片产生对应的数据;对于已删除的图片删除对应的数据
    """
    with open("StdPoseDatabase/Management.txt","r") as f:
        recording_list=f.readlines()
    current_image_list=os.listdir("StdPoseDatabase/Images")
    


