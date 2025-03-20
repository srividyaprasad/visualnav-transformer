import glob
import yaml
import copy
import cv2
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)


# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
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
subgoal = []
img_frame_id = 0

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
        cv2.imwrite(f"{OUTPUT_IMG_PATH}/out_{frame_id}", background)
        first_candidate = False

def callback_obs(msg):
    
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)
        img_frame_id +=1


def main(args: argparse.Namespace):
    global context_size

     # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    
     # load topomap
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node
    reached_goal = False

     # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

    print("Registered with master node. Waiting for image observations...")

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    # navigation loop
    while not rospy.is_shutdown():
        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            if model_params["model_type"] == "nomad":
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1) 
                obs_images = obs_images.to(device)
                mask = torch.zeros(1).long().to(device)  

                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in topomap[start:end + 1]]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                print("closest node:", closest_node)
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                    
                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    start_time = time.time()
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
                    print("time elapsed:", time.time() - start_time)

                naction = to_numpy(get_action(naction))

                all_actions_sampled = naction

                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
                print("published sampled actions")
                sampled_actions_pub.publish(sampled_actions_msg)
                naction = naction[0] 
                chosen_waypoint = naction[args.waypoint]

                transform_sampled_actions = transform_action_space(all_actions_sampled)
                transform_sampled_actions = check_validity(transform_sampled_actions)

                if context_queue:
                    last_image = context_queue[-1]
                    visualize_network_action(last_image,transform_sampled_actions, img_frame_id)
                else:
                    last_image = None 

            else:
                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                for i, sg_img in enumerate(topomap[start: end + 1]):
                    transf_obs_img = transform_images(context_queue, model_params["image_size"])
                    goal_data = transform_images(sg_img, model_params["image_size"])
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                    
                # predict distances and waypoints
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)
                # look for closest node
                min_dist_idx = np.argmin(distances)
                # chose subgoal and output waypoints
                if distances[min_dist_idx] > args.close_threshold:
                    chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                    closest_node = start + min_dist_idx
                else:
                    chosen_waypoint = waypoints[min(
                        min_dist_idx + 1, len(waypoints) - 1)][args.waypoint]
                    closest_node = min(start + min_dist_idx + 1, goal_node)
        # RECOVERY MODE
        if model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)  
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)
        reached_goal = closest_node == goal_node
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("Reached goal! Stopping...")
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
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


