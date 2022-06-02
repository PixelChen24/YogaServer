import numpy as np
import warnings
def get_score(standard_angle,measured_angle,weights):
    if sum(weights)!=1:
        warnings.warn("Weights settings not correct!")
    return weights*(1-abs(standard_angle-measured_angle)/standard_angle)


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