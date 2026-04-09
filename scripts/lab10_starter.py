#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import inf, sqrt, atan2, pi, sin, cos
from time import sleep, time
import queue
import json

import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Twist, Point32, PoseStamped, Pose, Vector3, Quaternion, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion

# AABB format: (x_min, x_max, y_min, y_max)
OBS_TYPE = Tuple[float, float, float, float]
# Position format: {"x": x, "y": y, "theta": theta}
POSITION_TYPE = Dict[str, float]

# don't change this
GOAL_THRESHOLD = 0.1


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


class PIDController:
    """
    Generates control action taking into account instantaneous error (proportional action),
    accumulated error (integral action) and rate of change of error (derivative action).
    """

    def __init__(self, kP, kI, kD, kS, u_min, u_max):
        assert u_min < u_max, "u_min should be less than u_max"
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.err_int = 0
        self.err_dif = 0
        self.err_prev = 0
        self.err_hist = []
        self.t_prev = 0
        self.u_min = u_min
        self.u_max = u_max

    def control(self, err, t):
        dt = t - self.t_prev
        self.err_hist.append(err)
        self.err_int += err
        if len(self.err_hist) > self.kS:
            self.err_int -= self.err_hist.pop(0)
        self.err_dif = err - self.err_prev
        u = (self.kP * err) + (self.kI * self.err_int * dt) + (self.kD * self.err_dif / dt)
        self.err_prev = err
        self.t_prev = t
        return max(self.u_min, min(u, self.u_max))


class Node:
    def __init__(self, position: POSITION_TYPE, parent: "Node"):
        self.position = position
        self.neighbors = []
        self.parent = parent

    def distance_to(self, other_node: "Node") -> float:
        return np.linalg.norm(self.position - other_node.position)

    def to_dict(self) -> Dict:
        return {"x": self.position[0], "y": self.position[1]}

    def __str__(self) -> str:
        return (
            f"Node<pos: {round(self.position[0], 4)}, {round(self.position[1], 4)}, #neighbors: {len(self.neighbors)}>"
        )


class RrtPlanner:

    def __init__(self, obstacles: List[OBS_TYPE], map_aabb: Tuple):
        self.obstacles = obstacles
        self.map_aabb = map_aabb
        self.graph_publisher = rospy.Publisher("/rrt_graph", MarkerArray, queue_size=10)
        self.plan_visualization_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        self.delta = 0.1
        self.obstacle_padding = 0.15
        self.goal_threshold = GOAL_THRESHOLD

    def visualize_plan(self, path: List[Dict]):
        marker_array = MarkerArray()
        for i, waypoint in enumerate(path):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position = Point(waypoint["x"], waypoint["y"], 0.0)
            marker.pose.orientation = Quaternion(0, 0, 0, 1)
            marker.scale = Vector3(0.075, 0.075, 0.1)
            marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.5)
            marker_array.markers.append(marker)
        self.plan_visualization_pub.publish(marker_array)

    def visualize_graph(self, graph: List[Node]):
        marker_array = MarkerArray()
        for i, node in enumerate(graph):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale = Vector3(0.05, 0.05, 0.05)
            marker.pose.position = Point(node.position[0], node.position[1], 0.01)
            marker.pose.orientation = Quaternion(0, 0, 0, 1)
            marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.5)
            marker_array.markers.append(marker)
        self.graph_publisher.publish(marker_array)

    def _randomly_sample_q(self) -> Node:
        # Choose uniform randomly sampled points
        ######### Your code starts here #########
        x_min, x_max, y_min, y_max = self.map_aabb
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return Node(np.array([x, y]), None)
        ######### Your code ends here #########

    def _nearest_vertex(self, graph: List[Node], q: Node) -> Node:
        # Determine vertex nearest to sampled point
        ######### Your code starts here #########
        nearest = None
        best_dist = inf
        for node in graph:
            d = np.linalg.norm(node.position - q.position)
            if d < best_dist:
                best_dist = d
                nearest = node
        assert nearest is not None
        return nearest
        ######### Your code ends here #########

    def _is_in_collision(self, q_rand: Node):
        x = q_rand.position[0]
        y = q_rand.position[1]
        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = obs
            x_min -= self.obstacle_padding
            y_min -= self.obstacle_padding
            x_max += self.obstacle_padding
            y_max += self.obstacle_padding
            if (x_min < x and x < x_max) and (y_min < y and y < y_max):
                return True
        return False

    def _extend(self, graph: List[Node], q_rand: Node):

        # Check if sampled point is in collision and add to tree if not
        ######### Your code starts here #########
        if self._is_in_collision(q_rand):
            return None

        q_near = self._nearest_vertex(graph, q_rand)
        direction = q_rand.position - q_near.position
        dist = np.linalg.norm(direction)
        if dist < 1e-9:
            return None

        step = min(self.delta, dist)
        q_new_pos = q_near.position + (direction / dist) * step
        q_new = Node(q_new_pos, q_near)

        # basic edge collision checking by discretizing along the segment
        n_checks = max(2, int(np.ceil(step / 0.02)))
        for alpha in np.linspace(0.0, 1.0, n_checks):
            p = q_near.position + alpha * (q_new.position - q_near.position)
            if self._is_in_collision(Node(p, None)):
                return None

        graph.append(q_new)
        q_near.neighbors.append(q_new)
        q_new.neighbors.append(q_near)
        return q_new
        ######### Your code ends here #########

    def generate_plan(self, start: POSITION_TYPE, goal: POSITION_TYPE) -> Tuple[List[POSITION_TYPE], List[Node]]:
        """Public facing API for generating a plan. Returns the plan and the graph.

        Return format:
            plan:
            [
                {"x": start["x"], "y": start["y"]},
                {"x": ...,      "y": ...},
                            ...
                {"x": goal["x"],  "y": goal["y"]},
            ]
            graph:
                [
                    Node<pos: x1, y1, #neighbors: n_1>,
                    ...
                    Node<pos: x_n, y_n, #neighbors: z>,
                ]
        """
        graph = [Node(np.array([start["x"], start["y"]]), None)]
        goal_node = Node(np.array([goal["x"], goal["y"]]), None)
        plan = []

        # Find path from start to goal location through tree
        ######### Your code starts here #########
        goal_reached: Optional[Node] = None

        for _ in range(self.max_iterations):
            q_rand = self._randomly_sample_q()
            q_new = self._extend(graph, q_rand)
            if q_new is None:
                continue

            if np.linalg.norm(q_new.position - goal_node.position) <= self.goal_threshold:
                goal_reached = q_new
                break

        if goal_reached is None:
            # no solution found within iteration budget; return a trivial plan to avoid crashing controller
            plan = [{"x": start["x"], "y": start["y"]}, {"x": goal["x"], "y": goal["y"]}]
            return plan, graph

        # Backtrack through parents to extract a path
        path_nodes: List[Node] = []
        cur = goal_reached
        while cur is not None:
            path_nodes.append(cur)
            cur = cur.parent
        path_nodes.reverse()

        plan = [{"x": n.position[0].item(), "y": n.position[1].item()} for n in path_nodes]
        # Ensure the final waypoint is exactly the goal location
        if np.linalg.norm(path_nodes[-1].position - goal_node.position) > 1e-6:
            plan.append({"x": goal["x"], "y": goal["y"]})
        else:
            plan[-1] = {"x": goal["x"], "y": goal["y"]}

        ######### Your code ends here #########
        return plan, graph


# Protip: copy the ObstacleFreeWaypointController class from lab5.py here
######### Your code starts here #########

def publish_waypoints(waypoints: List[Dict], publisher: rospy.Publisher):
    marker_array = MarkerArray()
    for i, waypoint in enumerate(waypoints):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = i
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = Point(waypoint["x"], waypoint["y"], 0.0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.5)
        marker_array.markers.append(marker)
    publisher.publish(marker_array)

class ObstacleFreeWaypointController:
    def __init__(self, waypoints: List[Dict]):
        self.waypoints = waypoints
        # Subscriber to the robot's current position (assuming you have Odometry data)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.waypoint_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        sleep(0.5)  # sleep to give time for rviz to subscribe to /waypoints
        publish_waypoints(self.waypoints, self.waypoint_pub)

        self.current_position = None

        # define linear and angular PID controllers here
        ######### Your code starts here #########
        self.linear_controller = PIDController(
            kP=0.5, kI=0.02, kD=0.1, kS=1.0,
            u_min=0.0, u_max=0.5,
        )
        self.angular_controller = PIDController(
            kP=0.5, kI=0.02, kD=0.45, kS=1.0,
            u_min=-2.84, u_max=2.84,
        )
        self.waypoint_tolerance = 0.1
        ######### Your code ends here #########

    def odom_callback(self, msg):
        # Extracting current position from Odometry message
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

    def calculate_error(self, goal_position: Dict) -> Optional[Tuple]:
        """Return distance and angle error between the current position and the provided goal_position. Returns None if
        the current position is not available.
        """
        if self.current_position is None:
            return None

        # Calculate error in position and orientation
        ######### Your code starts here #########
        dx = goal_position["x"] - self.current_position["x"]
        dy = goal_position["y"] - self.current_position["y"]
        distance_error = sqrt(dx * dx + dy * dy)
        angle_to_goal = atan2(dy, dx)
        angle_error = atan2(
            sin(angle_to_goal - self.current_position["theta"]),
            cos(angle_to_goal - self.current_position["theta"]),
        )
        ######### Your code ends here #########

        return distance_error, angle_error

    def control_robot(self):
        rate = rospy.Rate(20)  # 20 Hz
        ctrl_msg = Twist()

        # initialize first waypoint
        current_waypoint_idx = 0

        while not rospy.is_shutdown():

            # Travel through waypoints one at a time, checking if robot is close enough
            ######### Your code starts here #########
            if current_waypoint_idx >= len(self.waypoints):
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
                self.robot_ctrl_pub.publish(ctrl_msg)
                rate.sleep()
                continue

            goal = self.waypoints[current_waypoint_idx]
            err_tuple = self.calculate_error(goal)
            if err_tuple is None:
                rate.sleep()
                continue
            distance_error, angle_error = err_tuple

            if distance_error < self.waypoint_tolerance:
                current_waypoint_idx += 1
                ctrl_msg.linear.x = 0.0
                ctrl_msg.angular.z = 0.0
            else:
                t = time()
                ctrl_msg.linear.x = self.linear_controller.control(distance_error, t)
                ctrl_msg.angular.z = self.angular_controller.control(angle_error, t)
            self.robot_ctrl_pub.publish(ctrl_msg)
            ######### Your code ends here #########
            rate.sleep()

######### Your code ends here #########


""" Example usage

rosrun development lab10.py --map_filepath src/csci445l/scripts/lab10_map.json
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()
    with open(args.map_filepath, "r") as f:
        map_ = json.load(f)
        goal_position = map_["goal_position"]
        obstacles = map_["obstacles"]
        map_aabb = map_["map_aabb"]
        start_position = {"x": 0.0, "y": 0.0}

    rospy.init_node("rrt_planner")
    planner = RrtPlanner(obstacles, map_aabb)
    plan, graph = planner.generate_plan(start_position, goal_position)
    planner.visualize_plan(plan)
    planner.visualize_graph(graph)
    print(f"Plan: {plan}")


    controller = ObstacleFreeWaypointController(plan)

    try:
        while not rospy.is_shutdown():
            controller.control_robot()
    except rospy.ROSInterruptException:
        print("Shutting down...")
