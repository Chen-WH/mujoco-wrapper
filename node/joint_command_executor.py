#!/usr/bin/env python3
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class MujocoJointCommandExecutor(Node):
  def __init__(self):
    super().__init__('mujoco_joint_command_executor')

    self.declare_parameter('robot', 'ur10')
    self.declare_parameter('scene_xml', '')

    robot = self.get_parameter('robot').get_parameter_value().string_value
    scene_xml = self.get_parameter('scene_xml').get_parameter_value().string_value

    if scene_xml:
      self.xml_file = scene_xml
      self.n = 6
      self.default_command = [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]
    elif robot == 'ur10':
      self.n = 6
      self.xml_file = '/home/chenwh/ros2_ws/src/mujoco-wrapper/models/ur10/scene.xml'
      self.default_command = [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]
    elif robot == 'jaka_zu12':
      self.n = 6
      self.xml_file = '/home/chenwh/ros2_ws/src/mujoco-wrapper/models/jaka_zu12/scene.xml'
      self.default_command = [0.0, np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0]
    else:
      raise ValueError("Unsupported robot. Use 'ur10', 'jaka_zu12', or provide scene_xml.")

    self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
    self.create_subscription(JointTrajectory, '/joint_commands', self.command_callback, 10)

    self.latest_positions = list(self.default_command)
    self.latest_torques = [0.0] * self.n
    self.command_received = False

    self.get_logger().info(f'MuJoCo executor started with scene: {self.xml_file}')

  def command_callback(self, msg: JointTrajectory):
    if len(msg.points) == 0:
      return

    point = msg.points[0]
    if len(point.positions) >= self.n:
      self.latest_positions = [float(v) for v in point.positions[:self.n]]
      self.command_received = True
    if len(point.effort) >= self.n:
      self.latest_torques = [float(v) for v in point.effort[:self.n]]

  def publish_joint_states(self, data):
    msg = JointState()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.name = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    msg.position = [float(x) for x in data.qpos[:self.n]]
    msg.velocity = [float(x) for x in data.qvel[:self.n]]
    msg.effort = [float(x) for x in data.qfrc_actuator[:self.n]]
    self.joint_state_pub.publish(msg)

  def apply_command(self, data):
    if self.command_received:
      data.ctrl[:self.n] = self.latest_positions
    else:
      data.ctrl[:self.n] = self.default_command

  def run(self):
    model = mujoco.MjModel.from_xml_path(self.xml_file)
    data = mujoco.MjData(model)
    data.ctrl[:self.n] = self.default_command

    with mujoco.viewer.launch_passive(model, data) as viewer:
      while rclpy.ok():
        step_start = time.time()

        self.apply_command(data)
        mujoco.mj_step(model, data)
        viewer.sync()

        self.publish_joint_states(data)

        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
          time.sleep(remaining)

        rclpy.spin_once(self, timeout_sec=0.0)


if __name__ == '__main__':
  rclpy.init()
  node = MujocoJointCommandExecutor()

  try:
    node.run()
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()
