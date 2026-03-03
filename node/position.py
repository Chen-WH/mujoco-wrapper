#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import time
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from rosgraph_msgs.msg import Clock
import numpy as np

class MujocoNode(Node):
  def __init__(self):
    super().__init__('mujoco_ros2_node')
    self.rbt = input("Enter the robot name: (ur/jaka)")
    if self.rbt == 'ur':
      self.n = 6
      self.xml_file = '/home/chenwh/ros2_ws/src/mujoco-wrapper/models/ur10/scene.xml'
      self.position_target = [0.0, -np.pi/2, 0, -np.pi/2, 0, 0]
    elif self.rbt == 'jaka':
      self.n = 6
      self.xml_file = '/home/chenwh/ros2_ws/src/mujoco-wrapper/models/jaka_zu12/scene.xml'
      self.position_target = [0.0, np.pi/2, 0, np.pi/2, 0, 0]
    else:
      raise ValueError("Invalid robot name. Please enter 'ur' or 'jaka'.")
          
    self.paused = False
    self.PublishJointState = self.create_publisher(JointState,'/joint_state',10)
    self.PublishMujocoSimClock = self.create_publisher(Clock,'/clock',10)
    self.create_subscription(JointTrajectory, '/joint_trajectory', self.trajectory_callback, 10)
    self.create_subscription(JointState, '/joint_target', self.target_callback, 10)
    self.trajectory = []
    self.current_trajectory_index = 0
    self.counter = 0
    self.trajectory_received = False
    self.target_reached = False
    self.trajectory_dt = 0.008
    self.trajectory_hold_steps = 1
    self.position_command = self.position_target
    self.velocity_command = [0.0] * self.n
    self.torque_command = [0.0] * self.n
    self.position_state = [0.0] * self.n
    self.velocity_state = [0.0] * self.n
    self.torque_state = [0.0] * self.n

  def trajectory_callback(self, msg):
    self.trajectory = msg
    self.counter = 0
    self.current_trajectory_index = 0
    self.trajectory_received = True

    min_diff = float('inf')
    for i, point in enumerate(msg.points):
      diff = sum(abs(point.positions[j] - self.position_command[j]) for j in range(self.n))
      if diff < min_diff:
        min_diff = diff
        self.current_trajectory_index = i + 1
      else:
        break

  def target_callback(self, msg):
    self.position_target = msg.position

  def MujocoSim(self):
    model = mujoco.MjModel.from_xml_path(self.xml_file)
    data = mujoco.MjData(model)
    self.trajectory_hold_steps = max(1, int(round(self.trajectory_dt / model.opt.timestep)))
    with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
      while 1:
        step_start = time.time()

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        # Convert numpy arrays to plain Python lists of floats so ROS assertions pass
        joint_state_msg.position = [float(x) for x in data.qpos]
        self.position_state = joint_state_msg.position
        joint_state_msg.velocity = [float(x) for x in data.qvel]
        self.velocity_state = joint_state_msg.velocity
        joint_state_msg.effort = [float(x) for x in data.qfrc_actuator]
        self.torque_state = joint_state_msg.effort
        self.PublishJointState.publish(joint_state_msg)

        position_error = sum(abs(self.position_state[i] - self.position_target[i]) for i in range(self.n))
        if position_error < 1e-2:
          self.target_reached = True

        if self.trajectory_received and not self.target_reached:
          if self.current_trajectory_index < len(self.trajectory.points):
            point_command = self.trajectory.points[self.current_trajectory_index]
            self.position_command = point_command.positions
            self.velocity_command = point_command.velocities
            self.torque_command = point_command.effort

            data.ctrl[:] = self.position_command
            self.counter += 1
            if self.counter >= self.trajectory_hold_steps:
              self.current_trajectory_index += 1
              self.counter -= self.trajectory_hold_steps
          else:
            raise RuntimeError("New trajectory not received")
        else:
          data.ctrl[:] = self.position_target

        if not self.paused:
          mujoco.mj_step(model, data)
          viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        rclpy.spin_once(self, timeout_sec=0.0)

  def key_callback(self,keycode):
    if chr(keycode) == ' ':
      if self.paused == True:
        self.paused = False
      else :
        self.paused = True

if __name__ == '__main__':
  rclpy.init()
  mujoco_node = MujocoNode()

  try:
    mujoco_node.MujocoSim()
  except KeyboardInterrupt:
    pass
  finally:
    mujoco_node.destroy_node()
    rclpy.shutdown()