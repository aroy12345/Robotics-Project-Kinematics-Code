import numpy as np
from lib.calcJacobian import calcJacobian

def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities
    """
    J = calcJacobian(q_in)
    
    # Combine v_in and omega_in
    v_omega = np.concatenate([v_in, omega_in])
    
    # Create a mask for non-NaN values
    mask = ~np.isnan(v_omega)
    
    # Apply the mask to J and v_omega
    J_masked = J[mask]
    v_omega_masked = v_omega[mask]
    
    # Solve the least squares problem
    dq = np.linalg.lstsq(J_masked, v_omega_masked, rcond=None)[0]
    
    return dq.reshape(1, 7)

if __name__ == '__main__':
    # Example usage
    q_in = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    v_in = np.array([0.1, 0.2, np.nan])
    omega_in = np.array([np.nan, np.nan, 0.3])

    dq = IK_velocity(q_in, v_in, omega_in)
    print("Joint velocities:")
    print(dq)