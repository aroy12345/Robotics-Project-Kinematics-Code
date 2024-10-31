import numpy as np
from lib.calcJacobian import calcJacobian

def calcManipulability(q_in):
    """
    Calculate manipulability ellipsoid and index
    """
    # Calculate Jacobian
    J = calcJacobian(q_in)
    
    # Get position part of Jacobian (first 3 rows)
    J_pos = J[:3,:]
    
    # Calculate manipulability matrix
    M = J_pos @ J_pos.T
    
    # Calculate singular values
    U, s, Vh = np.linalg.svd(J_pos)
    
    # Calculate manipulability index (product of singular values)
    mu = np.prod(s)
    
    return mu, M