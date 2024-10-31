import numpy as np
from math import pi, acos
from scipy.linalg import null_space

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff
# from lib.IK_velocity import IK_velocity  #optional


class IK:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self,linear_tol=1e-4, angular_tol=1e-3, max_steps=1000, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        # Extract rotation matrices and positions
        R_target = target[:3, :3]
        R_current = current[:3, :3]
        p_target = target[:3, 3]
        p_current = current[:3, 3]
        
        # Calculate displacement (linear difference)
        displacement = p_target - p_current
        
        # Calculate rotation axis using calcAngDiff (similar to lab 2)
        axis = calcAngDiff(R_target, R_current)
        
        return displacement, axis
    @staticmethod
    def distance_and_angle(G, H):
        """
        Computes distance and angle between two transforms
        """
        # Calculate distance between origins
        distance = np.linalg.norm(G[:3, 3] - H[:3, 3])
        
        # Calculate angle using trace method
        R_diff = G[:3, :3].T @ H[:3, :3]
        trace = np.trace(R_diff)
        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        angle = np.arccos(cos_angle)
        
        return distance, angle
    def is_valid_solution(self,q,target):
        """
        Checks if solution meets joint limits and pose tolerances
        """
        # Check joint limits
        if not np.all(q >= self.lower) or not np.all(q <= self.upper):
            return False, "Joint limits violated"
        
        # Get current end effector pose
        _, current = self.fk.forward(q)
        
        # Check distance and angle tolerances
        distance, angle = self.distance_and_angle(target, current)
        
        if distance > self.linear_tol:
            return False, f"Linear tolerance not met: {distance} > {self.linear_tol}"
        if angle > self.angular_tol:
            return False, f"Angular tolerance not met: {angle} > {self.angular_tol}"
        
        return True, "Solution found within tolerances"

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q,target,method):
        """
        Computes joint velocity for primary task
        """
        # Get current end effector pose
        _, current = FK().forward(q)
        
        # Calculate displacement and axis
        displacement, axis = IK.displacement_and_axis(target, current)
        
        # Combine into desired end effector velocity
        v = np.concatenate([displacement, axis])
        
        # Get Jacobian
        J = calcJacobian(q)
        
        if method == 'J_pseudo':
            # Use pseudoinverse method
            J_pinv = np.linalg.pinv(J)
            dq = J_pinv @ v
        else:  # J_trans method
            # Use transpose method
            dq = J.T @ v
            
        return dq
    @staticmethod
    def joint_centering_task(q,rate=5e-1): 
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's angle and the center of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset # proportional term (implied quadratic cost)

        return dq
        
    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed, method, alpha):
        """
        Gradient descent solver for IK
        """
        q = seed
        rollout = []
        
        for step in range(self.max_steps):
            rollout.append(q)
            
            # Primary Task - End Effector Pose
            dq_ik = self.end_effector_task(q, target, method)
            
            # Secondary Task - Joint Centering
            dq_center = self.joint_centering_task(q)
            
            # Get Jacobian and its pseudoinverse for null space projection
            J = calcJacobian(q)
            J_pinv = np.linalg.pinv(J)
            
            # Null space projector
            N = np.eye(7) - J_pinv @ J
            
            # Combine tasks with null space projection
            dq = dq_ik + N @ dq_center
            
            # Scale step size
            dq = alpha * dq
            
            # Check termination conditions
            if np.linalg.norm(dq) < self.min_step_size:
                break
                
            # Update q
            q = q + dq
            
        success, message = self.is_valid_solution(q, target)
        return q, rollout, success, message
################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    ik = IK()

    # matches figure in the handout
    seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    target = np.array([
        [0,-1,0,-0.2],
        [-1,0,0,0],
        [0,0,-1,.5],
        [0,0,0, 1],
    ])

    # Using pseudo-inverse 
    q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target, seed, method='J_pseudo', alpha=.5)

    for i, q_pseudo in enumerate(rollout_pseudo):
        joints, pose = ik.fk.forward(q_pseudo)
        d, ang = IK.distance_and_angle(target,pose)
        print('iteration:',i,' q =',q_pseudo, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    # Using pseudo-inverse 
    q_trans, rollout_trans, success_trans, message_trans = ik.inverse(target, seed, method='J_trans', alpha=.5)

    for i, q_trans in enumerate(rollout_trans):
        joints, pose = ik.fk.forward(q_trans)
        d, ang = IK.distance_and_angle(target,pose)
        print('iteration:',i,' q =',q_trans, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    # compare
    print("\n method: J_pseudo-inverse")
    print("   Success: ",success_pseudo, ":  ", message_pseudo)
    print("   Solution: ",q_pseudo)
    print("   #Iterations : ", len(rollout_pseudo))
    print("\n method: J_transpose")
    print("   Success: ",success_trans, ":  ", message_trans)
    print("   Solution: ",q_trans)
    print("   #Iterations :", len(rollout_trans),'\n')
