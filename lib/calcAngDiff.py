import numpy as np

def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation 
    from the current orientation to the target orientation.
    
    :param R_des: 3x3 rotation matrix describing the desired orientation
    :param R_curr: 3x3 rotation matrix describing the current orientation
    :return: 3x1 vector of angular velocities [wx, wy, wz]
    """
    
    R_diff = R_curr.T @ R_des
     
    S = 0.5 * (R_diff - R_diff.T)
    
    temp = np.array([S[2, 1], S[0, 2], S[1, 0]])
    return R_curr @ temp
