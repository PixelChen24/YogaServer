import numpy as np
import warnings
def get_score(standard_angle,measured_angle,weights):
    if standard_angle.shape!=measured_angle.shape:
        warnings.warn("Fatal Error: StdAngle Size does not match Measured Size!")
        return 0
    if sum(weights)!=1:
        warnings.warn("Weights settings not correct!")
    chosen_index=np.where(weights>0)
    return sum(weights[chosen_index]*(1-abs(standard_angle[chosen_index]-measured_angle[chosen_index])/standard_angle[chosen_index]))


if __name__=="__main__":
    standard=np.array([
        120,30,90
    ])
    measure=np.array([
        105,35,75
    ])
    w=np.array([
        0.5,0.25,0.25
    ])
    print(get_score(standard_angle=standard, measured_angle=measure, weights=w))