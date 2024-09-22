import numpy as np
from math import pi, sin, cos

class FK():

    def __init__(self):
        self.dh_table = [
            [0,      -pi/2, 0.333,  0],
            [0,       pi/2, 0,      0],
            [0.0825,  pi/2, 0.316,  0],
            [-0.0825, -pi/2, 0,     0],
            [0,       pi/2, 0.384,  0],
            [0.088,   pi/2, 0,      0],
            [0,       0,    0.210,  -pi/4]
        ]
        self.joint_offsets = {
            2: np.array([0, 0, 0.195]),
            4: np.array([0, 0, 0.125]),
            5: np.array([0, 0, -0.015]),
            6: np.array([0, 0, 0.051])
        }
        self.base_elevation = 0.141  # Base elevation in meters
        
    def dh_transform(self, a, d, alpha, theta):
        """Compute the homogeneous transformation matrix based on DH parameters."""
        return np.array([
            [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0,           sin(alpha),             cos(alpha),            d           ],
            [0,           0,                      0,                     1           ]
        ])

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions - 8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
        T0e            - a 4 x 4 homogeneous transformation matrix, representing the end effector frame in the world frame
        """
        jointPositions = np.zeros((8, 3))
        T = np.eye(4)
        
        for i in range(7):  # For each joint
            a, alpha, d, theta_offset = self.dh_table[i]
            theta = q[i] + theta_offset
            Ti = self.dh_transform(a, d, alpha, theta)
            T = T @ Ti
            
            # Store the coordinate frame origin
            coord_frame_origin = T[:3, 3]
            
            # Apply joint offset if it exists
            if i+1 in self.joint_offsets:
                # Transform the local offset to world frame
                world_offset = T[:3, :3] @ self.joint_offsets[i+1]
                jointPositions[i+1] = coord_frame_origin + world_offset
            else:
                jointPositions[i+1] = coord_frame_origin

        # Add base elevation only to the first joint position
        jointPositions[0, 2] += self.base_elevation
        
        # The end-effector transformation doesn't change
        T0e = T

        return jointPositions, T0e

    def get_axis_of_rotation(self, q):
        raise NotImplementedError("This method is not implemented yet.")

    def compute_Ai(self, q):
        raise NotImplementedError("This method is not implemented yet.")

if __name__ == "__main__":
    fk = FK()
    q = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4])
    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n", joint_positions)
    print("End Effector Pose:\n", T0e)