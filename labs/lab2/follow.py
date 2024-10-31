import sys
import numpy as np
import rospy
from math import cos, sin, pi
import matplotlib.pyplot as plt
import geometry_msgs
import visualization_msgs
from tf.transformations import quaternion_from_matrix

from core.interfaces import ArmController
from core.utils import time_in_seconds

from lib.IK_velocity import IK_velocity
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff

#####################
## Rotation Helper ##
#####################

def rotvec_to_matrix(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-9:
        return np.eye(3)

    # Normalize to get rotation axis.
    k = rotvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


##################
## Follow class ##
##################

class JacobianDemo():
    """
    Demo class for testing Jacobian and Inverse Velocity Kinematics.
    Contains trajectories and controller callback function
    """
    active = False # When to stop commanding arm
    start_time = 0 # start time
    dt = 0.03 # constant for how to turn velocities into positions
    fk = FK()
    point_pub = rospy.Publisher('/vis/trace', geometry_msgs.msg.PointStamped, queue_size=10)
    ellipsoid_pub = rospy.Publisher('/vis/ellip', visualization_msgs.msg.Marker, queue_size=10)
    counter = 0
    x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position
    last_iteration_time = None


    ##################
    ## TRAJECTORIES ##
    ##################
    @staticmethod
    def ellipse(t, f=0.5, ry=0.15, rz=0.10):
        x0 = JacobianDemo.x0
        
        # Position
        y = ry * sin(2 * pi * f * t)
        z = rz * cos(2 * pi * f * t)
        xdes = x0 + np.array([0, y, z])
        
        # Velocity
        vy = 2 * pi * f * ry * cos(2 * pi * f * t)
        vz = -2 * pi * f * rz * sin(2 * pi * f * t)
        vdes = np.array([0, vy, vz])
        
        # Orientation (rotating around x-axis)
        ang = pi/4 * sin(2 * pi * f * t)
        r = ang * np.array([1.0, 0.0, 0.0])
        Rdes = rotvec_to_matrix(r)
        
        # Angular velocity
        ang_v = pi/4 * 2 * pi * f * cos(2 * pi * f * t)
        ang_vdes = ang_v * np.array([1.0, 0.0, 0.0])
        
        return Rdes, ang_vdes, xdes, vdes

    @staticmethod
    def line(t, f=1.0, L=0.15):
        x0 = JacobianDemo.x0
        
        # Position
        z = (L/2) * sin(2 * pi * f * t)
        xdes = x0 + np.array([0, 0, z])
        
        # Velocity
        vz = (L/2) * 2 * pi * f * cos(2 * pi * f * t)
        vdes = np.array([0, 0, vz])
        
        # Orientation (rotating around x-axis)
        ang = -np.pi + (np.pi/4.0) * sin(2 * pi * f * t)
        r = ang * np.array([1.0, 0.0, 0.0])
        Rdes = rotvec_to_matrix(r)
        
        # Angular velocity
        ang_v = (np.pi/4.0) * 2 * pi * f * cos(2 * pi * f * t)
        ang_vdes = ang_v * np.array([1.0, 0.0, 0.0])
        
        return Rdes, ang_vdes, xdes, vdes

    @staticmethod
    def eight(t, fx=0.5, fy=1.0, rx=0.15, ry=0.1):
        x0 = JacobianDemo.x0
        
        # Position (Lissajous Curve)
        xdes = x0 + np.array([rx * sin(2 * pi * fx * t), ry * sin(2 * pi * fy * t), 0])
        
        # Velocity
        vdes = np.array([
            rx * 2 * pi * fx * cos(2 * pi * fx * t),
            ry * 2 * pi * fy * cos(2 * pi * fy * t),
            0
        ])
        
        # Orientation (rotating around z-axis)
        ang = pi/4 * sin(2 * pi * (fx + fy) * t / 2)
        r = ang * np.array([0.0, 0.0, 1.0])
        Rdes = rotvec_to_matrix(r)
        
        # Angular velocity
        ang_v = pi/4 * 2 * pi * (fx + fy) / 2 * cos(2 * pi * (fx + fy) * t / 2)
        ang_vdes = ang_v * np.array([0.0, 0.0, 1.0])
        
        return Rdes, ang_vdes, xdes, vdes

    ###################
    ## VISUALIZATION ##
    ###################

    def show_ee_position(self):
        msg = geometry_msgs.msg.PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'endeffector'
        msg.point.x = 0
        msg.point.y = 0
        msg.point.z = 0
        self.point_pub.publish(msg)


    ################
    ## CONTROLLER ##
    ################

    def follow_trajectory(self, state, trajectory):

        if self.active:

            try:
                t = time_in_seconds() - self.start_time

                # get desired trajectory position and velocity
                Rdes, ang_vdes, xdes, vdes = trajectory(t)

                # get current end effector position
                q = state['position']

                joints, T0e = self.fk.forward(q)

                R = (T0e[:3,:3])
                x = (T0e[0:3,3])
                curr_x = np.copy(x.flatten())

                # First Order Integrator, Proportional Control with Feed Forward
                kp = 0.01
                v = vdes + kp * (xdes - curr_x)
                
                # Rotation
                kr = 0.01
                omega = ang_vdes + kr * calcAngDiff(Rdes, R).flatten()


                # Velocity Inverse Kinematics
                dq = IK_velocity(q, v, omega).flatten()


                # Get the correct timing to update with the robot
                if self.last_iteration_time == None:
                    self.last_iteration_time = time_in_seconds()

                self.dt = time_in_seconds() - self.last_iteration_time
                self.last_iteration_time = time_in_seconds()

                new_q = q + self.dt * dq

                arm.safe_set_joint_positions_velocities(new_q, dq)

                # Downsample visualization to reduce rendering overhead
                self.counter = self.counter + 1
                if self.counter == 10:
                    self.show_ee_position()
                    self.counter = 0

            except rospy.exceptions.ROSException:
                pass


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage:\n\tpython follow.py line\n\tpython follow.py ellipse\n\tpython follow.py eight")
        exit()

    rospy.init_node("follower")

    JD = JacobianDemo()

    if sys.argv[1] == 'line':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.line)
    elif sys.argv[1] == 'ellipse':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.ellipse)
    elif sys.argv[1] == 'eight':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.eight)
    else:
        print("invalid option")
        exit()

    arm = ArmController(on_state_callback=callback)

    # reset arm
    print("resetting arm...")
    arm.safe_move_to_position(arm.neutral_position())

    # q = np.array([ 0,    0,     0, 0,     0, pi, 0.75344866 ])
    # arm.safe_move_to_position(q)
    
    # start tracking trajectory
    JD.active = True
    JD.start_time = time_in_seconds()

    input("Press Enter to stop")
