import numpy as np
import warnings
body_angle=     ["左肩","右肩","左手肘","右手肘","左髋关节","右髋关节","左膝","右膝"]
body_suggestion=["左手大臂","右手大臂","左手小臂","右手小臂","左大腿","右大腿","左小腿","右小腿"]
excessive_suggestion=["向下收束一点","向下收束一点","向上抬一点","向上抬一点","向上抬一点","向上抬一点","向内收","向内收"]
lack_suggestion=     ["向上伸展一点","向上伸展一点","向下放一点","向下放一点","向内收一点","向内收一点","向外伸展","向外伸展"]
allow_error=10

def get_score(standard_angle,measured_angle,weights):
    if standard_angle.shape!=measured_angle.shape:
        warnings.warn("Fatal Error: StdAngle Size does not match Measured Size!")
        return 0
    if sum(weights)!=1:
        warnings.warn("Weights settings not correct!")
    chosen_index=np.where(weights>0)
    return sum(weights[chosen_index]*(1-abs(standard_angle[chosen_index]-measured_angle[chosen_index])/standard_angle[chosen_index]))


def get_suggestion(standard_angle,measured_angle,weights)->str:
    suggestions=[]
    if standard_angle.shape!=measured_angle.shape:
        warnings.warn("Fatal Error: StdAngle Size does not match Measured Size!")
        return "Error Input params"
    delta_angle=measured_angle-standard_angle
    delta_score=np.abs(delta_angle)/standard_angle*weights
    modify_angle=np.argsort(delta_score)[::-1]  # 优先调整差别最大造成分数下降最多的角度
    modifiy_first,modify_second=modify_angle[0],modify_angle[1]
    first_angle_suggestion=angle_suggestion(modifiy_first, delta_angle)
    second_angle_suggestion=angle_suggestion(modify_second, delta_angle)

    first_detail_suggestion=detail_suggestion(modifiy_first, delta_angle)
    second_detail_suggestion=detail_suggestion(modify_second, delta_angle)

    suggestions.append(first_angle_suggestion+first_detail_suggestion)
    suggestions.append(second_angle_suggestion+second_detail_suggestion)

    return first_angle_suggestion+first_detail_suggestion


def angle_suggestion(body_position,delta):
    suggestion=""
    if delta[body_position]>allow_error:
        suggestion = body_angle[body_position]+"角度偏大了 "
    elif delta[body_position]<-allow_error:
        suggestion = body_angle[body_position]+"角度偏小了 "
    return suggestion


def detail_suggestion(body_position,delta):
    suggestion=""
    if delta[body_position]>allow_error:
        suggestion = body_suggestion[body_position]+excessive_suggestion[body_position]+"才能更加标准哦"
    elif delta[body_position]<-allow_error:
        suggestion = body_suggestion[body_position]+lack_suggestion[body_position]+"才能更加标准哦"
    else:
        suggestion="做得非常棒！"
    return suggestion



if __name__=="__main__":
    standard=np.array([
        90,90,90,90,4564,16356,4651,456
    ])
    measure=np.array([
        100,120,56,70,45646,464,87,453
    ])
    w=np.array([
        0.25,0.25,0.25,0.25,0,0,0,0
    ])
    print(get_score(standard_angle=standard, measured_angle=measure, weights=w))
    print(get_suggestion(standard, measure, weights=w))