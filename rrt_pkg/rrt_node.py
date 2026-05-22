#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from ompl import base as ob
from ompl import geometric as og


from nav_msgs.msg import OccupancyGrid, Path
# i'm going to assume occupancy grid comes as the conventional nav_msgs type

from geometry_msgs.msg import PoseStamped
# assumption for init/goal poses


class RRTNode(Node):
    def __init__(self):
        super().__init__("rrt_node")

        self.path_pub = self.create_publisher(Path, "rrt_path", 10)


def main(args=None):
    rclpy.init(args=args)
    node = RRTNode()
    rclpy.spin(node)
    rclpy.shutdown()
