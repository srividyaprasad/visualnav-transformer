import argparse
import os
import shutil
from utils import msg_to_pil
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

IMAGE_TOPIC = "/sim_camera/rgb"
TOPOMAP_IMAGES_DIR = "../topomaps/images"
obs_img = None


def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


class TopomapCreatorNode(Node):
    def __init__(self, dir_path: str, dt: float):
        super().__init__('create_topomap')
        self.dir_path = dir_path
        self.dt = dt
        self.obs_img = None
        self.i = 0
        self.start_time = float("inf")

        # Subscriber to image topic
        self.create_subscription(Image, IMAGE_TOPIC, self.callback_obs, 10)
        # Publisher to subgoal topic
        self.subgoals_pub = self.create_publisher(Image, '/subgoals', 10)
        # Subscriber to joystick topic
        # self.create_subscription(Joy, "joy", self.callback_joy, 10)

        # Timer to check for inactivity
        self.create_timer(self.dt, self.timer_callback)

        self.topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, self.dir_path)
        if not os.path.isdir(self.topomap_name_dir):
            os.makedirs(self.topomap_name_dir)
        else:
            self.get_logger().info(f"{self.topomap_name_dir} already exists. Removing previous images...")
            remove_files_in_dir(self.topomap_name_dir)

    def callback_obs(self, msg: Image):
        global obs_img
        self.obs_img = msg_to_pil(msg)

    def timer_callback(self):
        global obs_img
        if self.obs_img is not None:
            self.obs_img.save(os.path.join(self.topomap_name_dir, f"{self.i}.png"))
            self.get_logger().info(f"Published image {self.i}")
            self.i += 1
            self.start_time = time.time()
            self.obs_img = None

        if time.time() - self.start_time > 2 * self.dt:
            self.get_logger().warning(f"Topic {IMAGE_TOPIC} not publishing anymore. Shutting down...")
            rclpy.shutdown()


def main(args: argparse.Namespace):
    rclpy.init()

    topomap_creator_node = TopomapCreatorNode(args.dir, args.dt)

    try:
        rclpy.spin(topomap_creator_node)
    except KeyboardInterrupt:
        pass
    finally:
        topomap_creator_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Code to generate topomaps from the {IMAGE_TOPIC} topic"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topological map images in ../topomaps/images directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.0,
        type=float,
        help=f"time between images sampled from the {IMAGE_TOPIC} topic (default: 1.0)",
    )
    args = parser.parse_args()

    main(args)
