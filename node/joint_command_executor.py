#!/usr/bin/env python3
import time
from typing import List

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class MujocoJointExecutor(Node):
    def __init__(self) -> None:
        super().__init__('mujoco_joint_executor_node')

        self.robot = self.declare_parameter('robot', 'ur').value
        self.command_topic = self.declare_parameter('command_topic', '/joint_commands').value
        self.joint_state_topic = self.declare_parameter('joint_state_topic', '/joint_states').value
        self.position_tolerance = float(self.declare_parameter('position_tolerance', 1e-2).value)

        if self.robot != 'ur':
            raise ValueError("Only robot='ur' is currently supported in this executor.")

        package_share = get_package_share_directory('mujoco-wrapper')
        self.xml_file = f"{package_share}/models/ur10/scene.xml"

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]
        self.n = len(self.joint_names)

        self.publish_joint_state = self.create_publisher(JointState, self.joint_state_topic, 20)
        self.create_subscription(JointTrajectory, self.command_topic, self.trajectory_callback, 10)

        self.trajectory: List[np.ndarray] = []
        self.trajectory_index = 0
        self.default_target = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0], dtype=float)
        self.position_target = self.default_target.copy()

        self.paused = False
        self.new_trajectory_received = False

        self.get_logger().info(
            f"MuJoCo executor ready. model={self.xml_file}, cmd={self.command_topic}, state={self.joint_state_topic}"
        )

    def trajectory_callback(self, msg: JointTrajectory) -> None:
        if len(msg.points) == 0:
            self.get_logger().warn('Received empty trajectory, ignored.')
            return

        new_traj: List[np.ndarray] = []
        for point in msg.points:
            if len(point.positions) < self.n:
                continue
            new_traj.append(np.asarray(point.positions[: self.n], dtype=float))

        if not new_traj:
            self.get_logger().warn('Received trajectory without valid positions, ignored.')
            return

        self.trajectory = new_traj
        self.trajectory_index = 0
        self.new_trajectory_received = True
        self.get_logger().info(f'Received /joint_commands with {len(self.trajectory)} points. Replanning execution queue reset.')

    def key_callback(self, keycode: int) -> None:
        if chr(keycode) == ' ':
            self.paused = not self.paused

    def run(self) -> None:
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)

        data.qpos[: self.n] = self.default_target
        data.ctrl[: self.n] = self.default_target

        with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
            while rclpy.ok():
                step_start = time.time()

                if self.new_trajectory_received:
                    self.new_trajectory_received = False

                if self.trajectory and self.trajectory_index < len(self.trajectory):
                    self.position_target = self.trajectory[self.trajectory_index]

                    error = float(np.sum(np.abs(np.asarray(data.qpos[: self.n]) - self.position_target)))
                    if error < self.position_tolerance:
                        self.trajectory_index += 1
                elif self.trajectory:
                    self.position_target = self.trajectory[-1]
                else:
                    self.position_target = self.default_target

                data.ctrl[: self.n] = self.position_target

                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.joint_names
                msg.position = [float(x) for x in data.qpos[: self.n]]
                msg.velocity = [float(x) for x in data.qvel[: self.n]]
                if data.qfrc_actuator.shape[0] >= self.n:
                    msg.effort = [float(x) for x in data.qfrc_actuator[: self.n]]
                self.publish_joint_state.publish(msg)

                if not self.paused:
                    mujoco.mj_step(model, data)
                    viewer.sync()

                rclpy.spin_once(self, timeout_sec=0.0)

                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0.0:
                    time.sleep(dt)


def main() -> None:
    rclpy.init()
    node = MujocoJointExecutor()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
