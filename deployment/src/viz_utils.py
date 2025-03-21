import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from datetime import datetime

OUTPUT_IMG_PATH = '../output_imgs'

output_video_path = f"{OUTPUT_IMG_PATH}/navigate_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.mp4"  # Output video file path
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
    # K = np.array([[ 1438.5902099609375, 0.0, 962.24627685546875], 
    #               [ 0.0, 1438.5902099609375, 722.44140625], 
    #               [ 0.0, 0.0, 1.0]], dtype=np.float32)
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