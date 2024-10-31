import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    Implements inverse velocity kinematics with a null space secondary task
    """
    # Calculate the Jacobian
    J = calcJacobian(q_in)
    
    # Combine v_in and omega_in for the primary task
    v_omega = np.concatenate([v_in, omega_in])
    
    # Create mask for non-NaN values
    mask = ~np.isnan(v_omega)
    
    # Get the masked Jacobian and velocity vector for primary task
    J_masked = J[mask]
    v_omega_masked = v_omega[mask].reshape(-1, 1)
    
    # Calculate the pseudoinverse of the masked Jacobian
    J_pinv = np.linalg.pinv(J_masked)
    
    # Calculate primary task joint velocities
    dq = J_pinv @ v_omega_masked
    
    # Calculate null space projector
    N = np.eye(7) - J_pinv @ J_masked
    
    # Calculate null space velocity
    null = (N @ b).T
    
    # Combine primary and null space velocities
    return dq.T + null


