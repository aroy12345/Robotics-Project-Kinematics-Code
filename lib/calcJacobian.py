import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """
    J = np.zeros((6, 7))
    fk = FK()
    
    # Get joint positions and end-effector position
    joint_positions, T0e = fk.forward(q_in)
    p_e = T0e[:3, 3]
    
    for i in range(7):
        # Get axis of rotation for joint i
        z_i = fk.get_axis_of_rotation(q_in, i)
        
        # Get position of joint i
        p_i = joint_positions[i]
        
        # Calculate linear velocity component
        J_v = np.cross(z_i, p_e - p_i)
        
        # Angular velocity component is simply the axis of rotation
        J_w = z_i
        
        # Combine to form the i-th column of the Jacobian
        J[:3, i] = J_v
        J[3:, i] = J_w
    
    return J

if __name__ == '__main__':
    q = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q), 3))