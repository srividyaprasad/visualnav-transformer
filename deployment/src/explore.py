import glob
import yaml
import copy
import cv2
import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml

# ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from PIL import Image as PILImage
import argparse
import time

# UTILS
from topic_names import (IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC)

# CONSTANTS
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]
OUTPUT_IMG_PATH = '../output_imgs'

# GLOBALS
context_queue = []
context_size = None
img_frame_id = 0

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

output_video_path = f"{OUTPUT_IMG_PATH}/output_video.mp4"  # Output video file path
frame_width = 1280  # Width of the frame
frame_height = 720  # Height of the frame
fps = 30  # Frames per second, adjust as necessary

# Define the codec and create the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 (you can use 'XVID', 'MJPG', etc.)
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

def transform_action_space(action):
    zero_mask = np.zeros((action.shape[0], action.shape[1], 1))
    action = np.concatenate((action, zero_mask), axis=-1)
    transform_actions = copy.deepcopy(action)
    transform_actions[:,:,2] = action[:,:,0]
    transform_actions[:,:,1] = action[:,:,2]
    transform_actions[:,:,0] = action[:,:,1]
    return transform_actions

def check_validity(actions):
    valid_actions = []
    for action in actions[:]:
        #check if action is non-decreasing
        valid = True
        for i in range(3):
            if not np.all(np.diff(action[:,i]) >= 0.0):
                valid = False
                break
        if valid:
            valid_actions.append(action)
    return np.array(valid_actions)


def visualize_network_action(img, actions, frame_id=0):

    if isinstance(img, PILImage.Image):  # Check if the input is a PIL image
        img = np.array(img)

    # --- Camera Intrinsic Parameters (example values) ---
    K = np.array([[ 617.54, 0.0, 318.655], 
                  [ 0.0, 617.5625, 244.1013], 
                  [ 0.0, 0.0, 1.0]], dtype=np.float32)
    distCoeffs = np.zeros((4, 1)) # Assuming no lens distortion

    # --- Placeholder for Camera Extrinsic Parameters ---
    rvec = np.zeros((3, 1), dtype=np.float32) # No rotation
    tvec = np.array([ 0.0, 0.55, -0.65], dtype=np.float32) # No translation
    background = img
    first_candidate = True
    for i, curve_points in enumerate(actions):
        projected_points, _ = cv2.projectPoints(curve_points, rvec, tvec, K, distCoeffs)
        projected_points = projected_points.reshape(-1, 2) # Reshape to (num_points, 2)
        # --- Draw the Curve on the Background Image ---
        # Draw small circles at each projected point
        for point in projected_points:
            cv2.circle(background, (int(point[0]), int(point[1])), radius=3, color=(0, 255, 255), thickness=-1)
        # Connect the points with a polyline
        pts = projected_points.reshape((-1, 1, 2)).astype(np.int32)
        line_color = (255, 0, 0)
        if first_candidate: line_color = (0, 0, 0)
        cv2.polylines(background, [pts], isClosed=False, color=line_color, thickness=2)
        text = f"Frame {frame_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        line_type = cv2.LINE_AA
        # Get the width and height of the text box
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Define a margin from the edges
        margin = 10
        # Calculate the bottom-left coordinate for the text so that it appears at the top-right
        # Note: The y-coordinate is the baseline of the text.
        position = (background.shape[1] - text_width - margin, text_height + margin)
        # Overlay the text onto the image in black color
        cv2.putText(background, text, position, font, font_scale, (0, 0, 0), thickness, line_type)
        # --- Save the Final Image to Disk ---
        # cv2.imwrite(f"{OUTPUT_IMG_PATH}/out_{frame_id}.png", background)
        video_writer.write(background)
        first_candidate = False

class ExplorationNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("exploration_node")
        self.args = args
        self.image_sub = self.create_subscription(Image, IMAGE_TOPIC, self.callback_obs, 10)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 10)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 10)
        self.timer = self.create_timer(1 / RATE, self.timer_callback)

    def callback_obs(self, msg):
        obs_img = msg_to_pil(msg)
        if context_size is not None:
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)
            global img_frame_id
            img_frame_id +=1

    def timer_callback(self):
        if len(context_queue) > context_size:
            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
            obs_images = obs_images.to(device)
            fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
            mask = torch.ones(1).long().to(device)  # ignore the goal

            # infer action
            with torch.no_grad():
                # encoder vision features
                obs_cond = model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)

                # (B, obs_horizon * obs_dim)
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(self.args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

                # initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (self.args.num_samples, model_params["len_traj_pred"], 2), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                start_time = self.get_clock().now()
                for k in noise_scheduler.timesteps[:]:
                    # predict noise
                    noise_pred = model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                elapsed_time = (self.get_clock().now() - start_time).nanoseconds / 1e9
                print("time elapsed:", elapsed_time)

            naction = to_numpy(get_action(naction)) # 8 x 8 x 2
            all_actions_sampled = naction
            
            flatten_naction = naction.flatten()

            sampled_actions_msg = Float32MultiArray()
            sampled_actions_msg.data = flatten_naction.tolist()
            self.sampled_actions_pub.publish(sampled_actions_msg)

            naction = naction[0]  # change this based on heuristic

            chosen_waypoint = naction[self.args.waypoint]

            if model_params["normalize"]:
                chosen_waypoint *= (MAX_V / RATE)

            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = chosen_waypoint.tolist()
            self.waypoint_pub.publish(waypoint_msg)
            print("Published waypoint")

            transform_sampled_actions = transform_action_space(all_actions_sampled)
            transform_sampled_actions = check_validity(transform_sampled_actions)
            if context_queue:
                last_image = context_queue[-1]
                global img_frame_id
                visualize_network_action(last_image,transform_sampled_actions, img_frame_id)
            else:
                last_image = None 

def main(args: argparse.Namespace):
    global context_size, model_params, num_diffusion_iters, noise_scheduler, device, model

    # Load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # Load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_params["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Initialize ROS 2 node
    rclpy.init(args=None)
    exploration_node = ExplorationNode(args)

    try:
        rclpy.spin(exploration_node)
    except KeyboardInterrupt:
        video_writer.release()
        pass
    finally:
        exploration_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,  # close waypoints exhibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)


