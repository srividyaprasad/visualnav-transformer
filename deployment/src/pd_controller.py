import numpy as np
import yaml
from typing import Tuple

# ROS 2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool, Float64MultiArray

from topic_names import (WAYPOINT_TOPIC, 
                         REACHED_GOAL_TOPIC)
from ros_data import ROSData
from utils import clip_angle

# CONSTS
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1/robot_config["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi/4

# GLOBALS
vel_msg = Twist()
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
reached_goal = False
reverse_mode = False
current_yaw = None

def clip_angle(theta) -> float:
	"""Clip angle to [-pi, pi]"""
	theta %= 2 * np.pi
	if -np.pi < theta < np.pi:
		return theta
	return theta - 2 * np.pi
      

def pd_controller(waypoint: np.ndarray) -> Tuple[float]:
	"""PD controller for the robot"""
	assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
	if len(waypoint) == 2:
		dx, dy = waypoint
	else:
		dx, dy, hx, hy = waypoint
	# this controller only uses the predicted heading if dx and dy near zero
	if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
		v = 0
		w = clip_angle(np.arctan2(hy, hx))/DT		
	elif np.abs(dx) < EPS:
		v =  0
		w = np.sign(dy) * np.pi/(2*DT)
	else:
		v = dx / DT
		w = np.arctan(dy/dx) / DT
	v = np.clip(v, 0, MAX_V)
	w = np.clip(w, -MAX_W, MAX_W)
	return v, w


def callback_drive(waypoint_msg: Float32MultiArray):
	"""Callback function for the waypoint subscriber"""
	global vel_msg
	print("seting waypoint")
	waypoint.set(waypoint_msg.data)
	
	
def callback_reached_goal(reached_goal_msg: Bool):
	"""Callback function for the reached goal subscriber"""
	global reached_goal
	reached_goal = reached_goal_msg.data

WHEELBASE = 0.75
WHEEL_RADIUS = 0.125 
HUB_WHEEL_RADIUS = 0.18 

def compute_hub_steering(v, w):
    if v > 0:
        v_hub = (
            np.hypot(v, w * WHEELBASE)
            / HUB_WHEEL_RADIUS
        )
    elif v < 0:
        v_hub = (
            -np.hypot(v, w * WHEELBASE)
            / HUB_WHEEL_RADIUS
        )
    else:
        # if the linear velocity is zero then it could be an in place turn
        v_hub = w * WHEELBASE / HUB_WHEEL_RADIUS

    if w != 0:
        # compute the radius of curvature
        r = v / w  # in m
        if r != 0:  # normal turn
            steering = np.arctan(WHEELBASE / r)  # in rad
        else:  # for in place turn we always turn right and then change the sign of the hub velocity
            steering = np.pi / 2  # in rad

    # Second case: it's a straight line
    else:
        steering = 0  # in rad
    
    return v_hub, steering

class PDControllerNode(Node):
    def __init__(self):
        super().__init__('pd_controller')
        
        # Subscribers
        self.waypoint_sub = self.create_subscription(
            Float32MultiArray,
            WAYPOINT_TOPIC,
            callback_drive,
            10  # Queue size
        )
        
        self.reached_goal_sub = self.create_subscription(
            Bool,
            REACHED_GOAL_TOPIC,
            callback_reached_goal,
            10  # Queue size
        )

        # Publisher
        self.vel_out = self.create_publisher(Twist, VEL_TOPIC, 10)
        
        self.drive_out_vhub = self.create_publisher(Float64MultiArray, "/vhub_command", 10)
        self.drive_out_steering = self.create_publisher(Float64MultiArray, "/steering_command", 10)

        # Timer for periodic execution (using ROS 2 timer)
        self.create_timer(1.0 / RATE, self.timer_callback)
        self.get_logger().info("PD Controller Node started.")

    def timer_callback(self):
        global vel_msg, reverse_mode
        if reached_goal:
            vel_msg = Twist()  # Stop the robot
            self.vel_out.publish(vel_msg)
            self.get_logger().info("Reached goal! Stopping...")
            rclpy.shutdown()  # Shutdown ROS 2 node
        elif waypoint.is_valid(verbose=True):
            v, w = pd_controller(waypoint.get())
            if reverse_mode:
                v *= -1
            vel_msg.linear.x = v
            vel_msg.angular.z = w
            self.get_logger().info(f"publishing new vel: {v}, {w}")
            self.vel_out.publish(vel_msg)

            v_hub, steering = compute_hub_steering(v, w)
            v_hub_msg = Float64MultiArray()
            v_hub_msg.data = [v_hub] # should be degree/s
            steering_msg = Float64MultiArray()
            steering_msg.data = [steering] # should be degree
            self.drive_out_vhub.publish(v_hub_msg)
            self.drive_out_steering.publish(steering_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PDControllerNode()
    rclpy.spin(node)  # Keep the node running
    rclpy.shutdown()

if __name__ == '__main__':
	main()
