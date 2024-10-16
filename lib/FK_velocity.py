import numpy as np
from lib.calcJacobian import calcJacobian

def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """
    # Calculate the Jacobian for the current configuration
    J = calcJacobian(q_in)
    
    # Convert dq to a column vector if it's not already
    dq = np.array(dq).reshape(7, 1)
    
    # Calculate the end-effector velocity
    velocity = J @ dq
    
    return velocity

if __name__ == '__main__':
    # Example usage
    q_in = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    dq = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    
    velocity = FK_velocity(q_in, dq)
    print("End-effector velocity:")
    print(velocity)