def match_orientations(v1_true, v2_true, v3_true, v1_est, v2_est, v3_est):
    import numpy as np
    import math
    v1_1 = move2firstquart(np.round(math.degrees(np.arccos(v1_true@v1_est)),3))
    v1_2 = move2firstquart(np.round(math.degrees(np.arccos(v1_true@v2_est)),3))
    v1_3 = move2firstquart(np.round(math.degrees(np.arccos(v1_true@v3_est)),3))
    v2_1 = move2firstquart(np.round(math.degrees(np.arccos(v2_true@v1_est)),3))
    v2_2 = move2firstquart(np.round(math.degrees(np.arccos(v2_true@v2_est)),3))
    v2_3 = move2firstquart(np.round(math.degrees(np.arccos(v2_true@v3_est)),3))

    idx = np.array([0,1,2,3,4,5,6,7,8,9,10]) #to avoid errors
    if v1_1<=v1_2 and v1_1<=v1_3: # v1_true = v1_est
        if v2_2<=v2_3:           # v2_true = v2_est
            idx = np.array([0,1,2,3,4,5,6,7,8,9,10])
        else:               # v2_true = v3_est
            idx = np.array([0,1,2,3,4,8,9,10,5,6,7])

    elif v1_2<=v1_1 and v1_2<=v1_3:   # v1_true = v2_est
        if v2_1<=v2_3:               # v2_true = v1_est
            idx = np.array([0,1,5,6,7,2,3,4,8,9,10])
        else:                   # v2_true = v3_est
            idx = np.array([0,1,5,6,7,8,9,10,2,3,4])

    elif v1_3<=v1_1 and v1_3<=v1_2:   # v1_true = v3_est
        if v2_1<=v2_2:               # v2_true = v1_est
            idx = np.array([0,1,8,9,10,2,3,4,5,6,7])
        else:                   # v2_true = v2_est
            idx = np.array([0,1,8,9,10,5,6,7,2,3,4])        

    return idx

def move2firstquart(angle):
    if angle>90:
        return  180-angle
    else:
        return angle