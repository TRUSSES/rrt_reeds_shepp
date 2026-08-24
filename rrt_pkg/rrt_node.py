#!/usr/bin/env python3

# rrt_node.py
# Original author: Benjamin Aziel
# Modified by: Jiaming Chen

# rrt connect with reeds-shepp state space

import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from rrt_pkg.ompl_planner import RRTPlanner
from rrt_pkg.planner import Grid, VehicleFootprint, yaw_from_quaternion

try:
    from trusses_custom_interfaces.msg import ExtrapolatedMap
except ImportError:
    ExtrapolatedMap = None

DEFAULT_TURNING_RADIUS = 0.5
DEFAULT_PLAN_TIMEOUT = 5.0
STATUS_WAITING_FOR_MAP = "waiting_for_map"
STATUS_WAITING_FOR_CURRENT_POSE = "waiting_for_current_pose"
STATUS_WAITING_FOR_GOAL_POSE = "waiting_for_goal_pose"
STATUS_PLANNING = "planning"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_INVALID_START = "invalid_start"
STATUS_INVALID_GOAL = "invalid_goal"


class RRTNode(Node):
    def __init__(self):
        super().__init__("rrt_node")

        self.occ_grid = None
        self.pose_current = None
        self.pose_goal = None
        self.bounds = None
        self.map_frame_id = "map"
        self.declare_parameters()
        self.load_parameters()

        self.create_subscriptions()
        self.create_publishers()
        self.publish_status(STATUS_WAITING_FOR_MAP)

    def declare_parameters(self):
        self.declare_parameter("turning_radius", DEFAULT_TURNING_RADIUS)
        self.declare_parameter("planning_timeout", DEFAULT_PLAN_TIMEOUT)
        self.declare_parameter("planner_type", "rrt_connect")
        self.declare_parameter("collision_check_step", 0.05)
        self.declare_parameter("occupancy_threshold", 50)
        self.declare_parameter("unknown_is_occupied", True)
        self.declare_parameter("footprint_type", "rectangle")
        self.declare_parameter("robot_radius", 0.25)
        self.declare_parameter("vehicle_length", 0.7)
        self.declare_parameter("vehicle_width", 0.45)
        self.declare_parameter("safety_margin", 0.05)
        self.declare_parameter("risk_map_topic", "extrapolated_map")
        self.declare_parameter("traversability_threshold", 0.0)

    def load_parameters(self):
        self.turning_radius = float(self.get_parameter("turning_radius").value)
        self.plan_timeout = float(self.get_parameter("planning_timeout").value)
        self.planner_type = str(self.get_parameter("planner_type").value)
        self.collision_check_step = float(
            self.get_parameter("collision_check_step").value
        )
        self.occupancy_threshold = int(
            self.get_parameter("occupancy_threshold").value
        )
        self.unknown_is_occupied = bool(
            self.get_parameter("unknown_is_occupied").value
        )
        self.footprint_type = str(self.get_parameter("footprint_type").value).lower()
        self.robot_radius = float(self.get_parameter("robot_radius").value)
        self.vehicle_length = float(self.get_parameter("vehicle_length").value)
        self.vehicle_width = float(self.get_parameter("vehicle_width").value)
        self.safety_margin = float(self.get_parameter("safety_margin").value)
        self.risk_map_topic = str(self.get_parameter("risk_map_topic").value)
        self.traversability_threshold = float(
            self.get_parameter("traversability_threshold").value
        )
        self.validate_parameters()

        self.footprint = None
        if self.footprint_type == "rectangle":
            self.footprint = VehicleFootprint(
                self.vehicle_length,
                self.vehicle_width,
            )

    def validate_parameters(self):
        positive = {
            "turning_radius": self.turning_radius,
            "planning_timeout": self.plan_timeout,
            "collision_check_step": self.collision_check_step,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"parameter '{name}' must be greater than zero")
        if not 0 <= self.occupancy_threshold <= 100:
            raise ValueError("parameter 'occupancy_threshold' must be in [0, 100]")
        if self.footprint_type not in {"point", "circle", "rectangle"}:
            raise ValueError(
                "parameter 'footprint_type' must be point, circle, or rectangle"
            )
        if self.robot_radius < 0.0 or self.safety_margin < 0.0:
            raise ValueError("robot_radius and safety_margin cannot be negative")
        if self.footprint_type == "circle" and self.robot_radius <= 0.0:
            raise ValueError("robot_radius must be positive for a circle footprint")
        if self.footprint_type == "rectangle" and (
            self.vehicle_length <= 0.0 or self.vehicle_width <= 0.0
        ):
            raise ValueError(
                "vehicle_length and vehicle_width must be positive "
                "for a rectangle footprint"
            )

    def create_subscriptions(self):
        self.map_sub = self.create_subscription(
            OccupancyGrid, "map", self.map_callback, 10
        )
        self.risk_map_sub = None
        if ExtrapolatedMap is not None:
            self.risk_map_sub = self.create_subscription(
                ExtrapolatedMap,
                self.risk_map_topic,
                self.risk_map_callback,
                10,
            )
        else:
            self.get_logger().warning(
                "trusses_custom_interfaces is unavailable; "
                "the extrapolated risk-map subscription is disabled"
            )
        self.pose_sub = self.create_subscription(
            PoseStamped, "pose_current", self.pose_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, "pose_goal", self.goal_callback, 10
        )

    def create_publishers(self):
        self.path_pub = self.create_publisher(Path, "rrt_path", 10)
        self.status_pub = self.create_publisher(String, "rrt_status", 10)

    def map_callback(self, msg: OccupancyGrid):
        self.occ_grid = Grid.from_occupancy_grid_msg(
            msg,
            occupancy_threshold=self.occupancy_threshold,
            unknown_is_occupied=self.unknown_is_occupied,
            inflation_radius=self.inflation_radius,
        )
        self.bounds = self.occ_grid.planning_bounds()
        self.map_frame_id = self.occ_grid.frame_id
        self.get_logger().info("map received")
        self.try_plan()

    def risk_map_callback(self, msg):
        try:
            self.occ_grid = Grid.from_extrapolated_map_msg(
                msg,
                traversability_threshold=self.traversability_threshold,
                inflation_radius=self.inflation_radius,
            )
        except ValueError as exc:
            self.get_logger().error(f"invalid extrapolated risk map: {exc}")
            return
        self.bounds = self.occ_grid.planning_bounds()
        self.map_frame_id = self.occ_grid.frame_id
        self.get_logger().info("extrapolated risk map received")
        self.try_plan()

    def pose_callback(self, msg: PoseStamped):
        self.pose_current = msg
        self.try_plan()

    def goal_callback(self, msg: PoseStamped):
        self.pose_goal = msg
        self.try_plan()

    # replan on every map, pose, and goal update
    def try_plan(self):
        missing_status = self.missing_input_status()
        if missing_status is not None:
            self.publish_status(missing_status)
            return

        self.publish_status(STATUS_PLANNING)
        start = (
            self.pose_current.pose.position.x,
            self.pose_current.pose.position.y,
            self.yaw_from_pose(self.pose_current),
        )
        goal = (
            self.pose_goal.pose.position.x,
            self.pose_goal.pose.position.y,
            self.yaw_from_pose(self.pose_goal),
        )
        if not self.occ_grid.pose_is_free(*start, footprint=self.footprint):
            self.publish_status(STATUS_INVALID_START)
            self.get_logger().error("start pose is outside the free map area")
            return
        if not self.occ_grid.pose_is_free(*goal, footprint=self.footprint):
            self.publish_status(STATUS_INVALID_GOAL)
            self.get_logger().error("goal pose is outside the free map area")
            return

        planner = RRTPlanner(
            self.occ_grid,
            turning_radius=self.turning_radius,
            bounds=self.bounds,
            planner_type=self.planner_type,
            collision_check_step=self.collision_check_step,
            footprint=self.footprint,
        )
        result = planner.plan(start, goal, timeout=self.plan_timeout)

        if result.path is None:
            self.publish_status(STATUS_FAILED)
            self.get_logger().error("planning failed")
            return

        self.publish_status(STATUS_SUCCEEDED)
        self.get_logger().info(f"found path with {len(result.path)} states")
        self.path_pub.publish(self.make_path_msg(result.path))

    def missing_input_status(self):
        if self.occ_grid is None or self.bounds is None:
            return STATUS_WAITING_FOR_MAP
        if self.pose_current is None:
            return STATUS_WAITING_FOR_CURRENT_POSE
        if self.pose_goal is None:
            return STATUS_WAITING_FOR_GOAL_POSE
        return None

    @property
    def inflation_radius(self):
        if self.footprint_type == "circle":
            return self.robot_radius + self.safety_margin
        return self.safety_margin

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def yaw_from_pose(self, msg):
        return yaw_from_quaternion(msg.pose.orientation)

    def set_pose_yaw(self, pose, yaw):
        pose.orientation.z = math.sin(yaw / 2.0)
        pose.orientation.w = math.cos(yaw / 2.0)

    def make_path_msg(self, path):
        msg = Path()
        msg.header.frame_id = self.map_frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y, yaw in path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            self.set_pose_yaw(pose.pose, yaw)
            msg.poses.append(pose)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = RRTNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
