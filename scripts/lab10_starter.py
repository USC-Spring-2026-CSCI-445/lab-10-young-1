#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import inf, sqrt, atan2, pi
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
        min_dist = inf
        for node in graph:
            d = node.distance_to(q)
            if d < min_dist:
                min_dist = d
                nearest = node
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
            return

        q_near = self._nearest_vertex(graph, q_rand)
        dist = q_near.distance_to(q_rand)

        if dist <= self.delta:
            q_new = q_rand
        else:
            direction = (q_rand.position - q_near.position) / dist
            new_pos = q_near.position + self.delta * direction
            q_new = Node(new_pos, None)

        if self._is_in_collision(q_new):
            return

        q_new.parent = q_near
        q_near.neighbors.append(q_new)
        q_new.neighbors.append(q_near)
        graph.append(q_new)
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
        K = 5000  # number of iterations — vary for demos

        for i in range(K):
            q_rand = self._randomly_sample_q()
            self._extend(graph, q_rand)

            # check if the newest node reached the goal
            if len(graph) > 1:
                newest = graph[-1]
                if newest.distance_to(goal_node) < self.goal_threshold:
                    # backtrack through parents to extract path
                    current = newest
                    while current is not None:
                        plan.append(current.to_dict())
                        current = current.parent
                    plan.reverse()
                    plan.append({"x": goal["x"], "y": goal["y"]})
                    print(f"Path found after {i + 1} iterations!")
                    return plan, graph

        print("No path found within iteration limit.")
        ######### Your code ends here #########
        return plan, graph


# Protip: copy the ObstacleFreeWaypointController class from lab5.py here
######### Your code starts here #########
class ObstacleFreeWaypointController:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.current_wp_index = 0
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_callback)
        self.current_pose = None
        self.linear_pid = PIDController(kP=0.5, kI=0.0, kD=0.1, kS=10, u_min=0.0, u_max=0.2)
        self.angular_pid = PIDController(kP=1.0, kI=0.0, kD=0.1, kS=10, u_min=-1.5, u_max=1.5)

    def _odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def control_robot(self):
        if self.current_pose is None or self.current_wp_index >= len(self.waypoints):
            return

        wp = self.waypoints[self.current_wp_index]
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        quat = self.current_pose.orientation
        _, _, theta = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        dx = wp["x"] - x
        dy = wp["y"] - y
        dist = sqrt(dx**2 + dy**2)

        if dist < GOAL_THRESHOLD:
            self.current_wp_index += 1
            if self.current_wp_index >= len(self.waypoints):
                self.cmd_pub.publish(Twist())
                print("Goal reached!")
            return

        desired_angle = atan2(dy, dx)
        angle_err = desired_angle - theta
        angle_err = atan2(np.sin(angle_err), np.cos(angle_err))

        t = time()
        twist = Twist()
        twist.angular.z = self.angular_pid.control(angle_err, t)
        twist.linear.x = self.linear_pid.control(dist, t) if abs(angle_err) < pi / 4 else 0.0
        self.cmd_pub.publish(twist)
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
    controller = ObstacleFreeWaypointController(plan)

    try:
        while not rospy.is_shutdown():
            controller.control_robot()
    except rospy.ROSInterruptException:
        print("Shutting down...")