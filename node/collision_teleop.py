#!/usr/bin/env python3
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class MujocoCollisionTeleopNode(Node):
  def __init__(self):
    super().__init__('mujoco_collision_teleop_node')

    self.n = 6
    self.xml_file = '/home/chenwh/ros2_ws/src/mujoco-wrapper/models/reorientation_cube/ur10_collision_scene.xml'
    self.initial_ctrl = [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]

    self.joint_state_pub = self.create_publisher(JointState, '/joint_state', 10)
    self.create_subscription(JointTrajectory, '/joint_trajectory', self.trajectory_callback, 10)

    self.trajectory = None
    self.current_trajectory_index = 0
    self.counter = 0
    self.trajectory_received = False
    self.last_pose_print_time = 0.0

  def resolve_ee_body_id(self, model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wrist_3_link')

  def trajectory_callback(self, msg: JointTrajectory):
    self.trajectory = msg
    self.current_trajectory_index = 0
    self.counter = 0
    self.trajectory_received = True

  def publish_joint_state(self, data):
    msg = JointState()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.position = [float(x) for x in data.qpos[:self.n]]
    msg.velocity = [float(x) for x in data.qvel[:self.n]]
    msg.effort = [float(x) for x in data.qfrc_actuator[:self.n]]
    self.joint_state_pub.publish(msg)

  def apply_joint_command(self, data):
    if self.trajectory_received and self.trajectory is not None and len(self.trajectory.points) > 0:
      if self.counter == 0 and self.current_trajectory_index < len(self.trajectory.points):
        point = self.trajectory.points[self.current_trajectory_index]
        if len(point.positions) >= self.n:
          data.ctrl[:self.n] = [float(v) for v in point.positions[:self.n]]
        self.current_trajectory_index += 1

      self.counter += 1
      if self.counter >= 4:
        self.counter = 0

      if self.current_trajectory_index >= len(self.trajectory.points):
        self.current_trajectory_index = len(self.trajectory.points) - 1
    else:
      data.ctrl[:self.n] = self.initial_ctrl

  def run(self):
    model = mujoco.MjModel.from_xml_path(self.xml_file)
    data = mujoco.MjData(model)
    ee_body_id = self.resolve_ee_body_id(model)

    data.ctrl[:self.n] = self.initial_ctrl

    self.get_logger().info('Collision teleop started with obstacle spheres visible.')
    with mujoco.viewer.launch_passive(model, data) as viewer:
      while rclpy.ok():
        step_start = time.time()

        self.publish_joint_state(data)
        self.apply_joint_command(data)

        mujoco.mj_step(model, data)
        viewer.sync()

        now_time = time.time()
        if now_time - self.last_pose_print_time >= 1.0:
          pos = data.xpos[ee_body_id]
          quat = data.xquat[ee_body_id]
          self.get_logger().info(
              'EE pose | quat[wxyz]=[%.5f, %.5f, %.5f, %.5f], pos[xyz]=[%.5f, %.5f, %.5f]'
              % (quat[0], quat[1], quat[2], quat[3], pos[0], pos[1], pos[2]))
          self.last_pose_print_time = now_time

        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
          time.sleep(remaining)

        rclpy.spin_once(self, timeout_sec=0.0)


if __name__ == '__main__':
  rclpy.init()
  node = MujocoCollisionTeleopNode()

  try:
    node.run()
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()