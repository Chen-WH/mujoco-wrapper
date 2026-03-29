#!/usr/bin/env python3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


@dataclass(frozen=True)
class RobotConfig:
    robot: str
    scene_relative_path: str
    joint_state_names: List[str]
    mujoco_joint_names: List[str]
    actuator_names: List[str]
    default_target: np.ndarray
    control_mode: str
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray


def _array(values: List[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


ROBOT_CONFIGS: Dict[str, RobotConfig] = {
    'ur': RobotConfig(
        robot='ur',
        scene_relative_path='models/ur10/scene.xml',
        joint_state_names=[
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ],
        mujoco_joint_names=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        actuator_names=['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6'],
        default_target=_array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]),
        control_mode='direct',
        kp=np.zeros(6, dtype=float),
        kd=np.zeros(6, dtype=float),
        effort_limit=np.full(6, np.inf, dtype=float),
    ),
    'franka': RobotConfig(
        robot='franka',
        scene_relative_path='models/franka_panda/scene.xml',
        joint_state_names=[
            'panda_joint1',
            'panda_joint2',
            'panda_joint3',
            'panda_joint4',
            'panda_joint5',
            'panda_joint6',
            'panda_joint7',
        ],
        mujoco_joint_names=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'],
        actuator_names=['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7'],
        default_target=np.zeros(7, dtype=float),
        control_mode='impedance_torque',
        #kp=_array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0]),
        #kd=_array([45.0, 45.0, 35.0, 35.0, 20.0, 20.0, 20.0]),
        kp=_array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0]),
        kd=_array([ 450.0,  450.0,  350.0,  350.0,  200.0,  200.0,  200.0]),
        effort_limit=_array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
    ),
    'leap_left': RobotConfig(
        robot='leap_left',
        scene_relative_path='models/leap_hand/scene_left.xml',
        joint_state_names=[str(i) for i in range(16)],
        mujoco_joint_names=[
            'if_mcp', 'if_rot', 'if_pip', 'if_dip',
            'mf_mcp', 'mf_rot', 'mf_pip', 'mf_dip',
            'rf_mcp', 'rf_rot', 'rf_pip', 'rf_dip',
            'th_cmc', 'th_axl', 'th_mcp', 'th_ipl',
        ],
        actuator_names=[
            'if_mcp_act', 'if_rot_act', 'if_pip_act', 'if_dip_act',
            'mf_mcp_act', 'mf_rot_act', 'mf_pip_act', 'mf_dip_act',
            'rf_mcp_act', 'rf_rot_act', 'rf_pip_act', 'rf_dip_act',
            'th_cmc_act', 'th_axl_act', 'th_mcp_act', 'th_ipl_act',
        ],
        default_target=np.zeros(16, dtype=float),
        control_mode='direct',
        kp=np.zeros(16, dtype=float),
        kd=np.zeros(16, dtype=float),
        effort_limit=np.full(16, np.inf, dtype=float),
    ),
}


class MujocoJointExecutor(Node):
    def __init__(self) -> None:
        super().__init__('mujoco_joint_executor_node')

        self.robot = str(self.declare_parameter('robot', 'ur').value)
        if self.robot not in ROBOT_CONFIGS:
            raise ValueError(f"Unsupported robot='{self.robot}'. Expected one of {sorted(ROBOT_CONFIGS.keys())}.")
        self.config = ROBOT_CONFIGS[self.robot]

        self.command_topic = self.declare_parameter('command_topic', '/joint_commands').value
        self.joint_state_topic = self.declare_parameter('joint_state_topic', '/joint_states').value
        self.position_tolerance = float(self.declare_parameter('position_tolerance', 1e-2).value)
        self.xml_file_param = str(self.declare_parameter('xml_file', '').value)
        self.mass_scale = float(self.declare_parameter('mass_scale', 1.0).value)
        self.payload_mass = float(self.declare_parameter('payload_mass', 0.0).value)
        self.payload_body_name = str(self.declare_parameter('payload_body_name', 'attachment').value)
        self.payload_com = np.asarray(
            self.declare_parameter('payload_com', [0.0, 0.0, 0.05]).value, dtype=float
        )
        if self.payload_com.shape != (3,):
            raise ValueError('payload_com must contain exactly 3 values.')

        package_share = get_package_share_directory('mujoco-wrapper')
        self.xml_file = self.xml_file_param or f"{package_share}/{self.config.scene_relative_path}"

        self.joint_names = list(self.config.joint_state_names)
        self.mujoco_joint_names = list(self.config.mujoco_joint_names)
        self.actuator_names = list(self.config.actuator_names)
        self.n = len(self.joint_names)

        self.publish_joint_state = self.create_publisher(JointState, self.joint_state_topic, 20)
        self.create_subscription(JointTrajectory, self.command_topic, self.trajectory_callback, 10)

        self.trajectory: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self.trajectory_start_time: Optional[float] = None
        self.default_target = self.config.default_target.copy()
        self.position_target = self.default_target.copy()
        self.velocity_target = np.zeros(self.n, dtype=float)
        self.effort_target = np.zeros(self.n, dtype=float)

        self.paused = False
        self.qpos_addrs: List[int] = []
        self.qvel_addrs: List[int] = []
        self.ctrl_addrs: List[int] = []

        self.get_logger().info(
            f"MuJoCo executor ready. robot={self.robot}, mode={self.config.control_mode}, "
            f"model={self.xml_file}, cmd={self.command_topic}, state={self.joint_state_topic}, "
            f"mass_scale={self.mass_scale:.3f}, payload_mass={self.payload_mass:.3f}"
        )

    def _lookup_body_id(self, model: mujoco.MjModel, body_name: str) -> int:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in {self.xml_file}")
        return int(body_id)

    def _apply_runtime_model_variations(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        dirty = False

        if self.robot == 'ur' and abs(self.mass_scale - 1.0) > 1e-9:
            for body_name in [
                'shoulder_link',
                'upper_arm_link',
                'forearm_link',
                'wrist_1_link',
                'wrist_2_link',
                'wrist_3_link',
            ]:
                body_id = self._lookup_body_id(model, body_name)
                model.body_mass[body_id] *= self.mass_scale
                model.body_inertia[body_id] *= self.mass_scale
            dirty = True
            self.get_logger().info(f'Applied UR10 mass/inertia scale {self.mass_scale:.3f} to MuJoCo plant.')

        if self.payload_mass > 0.0:
            body_id = self._lookup_body_id(model, self.payload_body_name)
            model.body_mass[body_id] = self.payload_mass
            model.body_inertia[body_id] = np.full(3, max(1e-6, 1e-6 * self.payload_mass), dtype=float)
            model.body_ipos[body_id] = self.payload_com
            dirty = True
            self.get_logger().info(
                f"Attached payload mass={self.payload_mass:.3f}kg com={self.payload_com.tolist()} "
                f"to body '{self.payload_body_name}'."
            )

        if dirty:
            mujoco.mj_setConst(model, data)

    def _lookup_joint_addresses(self, model: mujoco.MjModel) -> None:
        self.qpos_addrs = []
        self.qvel_addrs = []
        for name in self.mujoco_joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in {self.xml_file}")
            self.qpos_addrs.append(int(model.jnt_qposadr[joint_id]))
            self.qvel_addrs.append(int(model.jnt_dofadr[joint_id]))

        self.ctrl_addrs = []
        for name in self.actuator_names:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in {self.xml_file}")
            self.ctrl_addrs.append(int(actuator_id))

    def _joint_vector_from_data(self, data: mujoco.MjData, addrs: List[int], source: np.ndarray) -> np.ndarray:
        return np.asarray([float(source[idx]) for idx in addrs], dtype=float)

    def _build_command_from_msg(self, msg: JointTrajectory, point_index: int) -> Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        point = msg.points[point_index]
        if msg.joint_names:
            name_to_idx = {name: idx for idx, name in enumerate(msg.joint_names)}
            if not all(name in name_to_idx for name in self.joint_names):
                return None
            if any(name_to_idx[name] >= len(point.positions) for name in self.joint_names):
                return None
            positions = np.asarray([point.positions[name_to_idx[name]] for name in self.joint_names], dtype=float)
            if len(point.velocities) >= len(msg.joint_names):
                velocities = np.asarray([point.velocities[name_to_idx[name]] for name in self.joint_names], dtype=float)
            else:
                velocities = np.zeros(self.n, dtype=float)
            if len(point.effort) >= len(msg.joint_names):
                efforts = np.asarray([point.effort[name_to_idx[name]] for name in self.joint_names], dtype=float)
            else:
                efforts = np.zeros(self.n, dtype=float)
        else:
            if len(point.positions) < self.n:
                return None
            positions = np.asarray(point.positions[: self.n], dtype=float)
            if len(point.velocities) >= self.n:
                velocities = np.asarray(point.velocities[: self.n], dtype=float)
            else:
                velocities = np.zeros(self.n, dtype=float)
            if len(point.effort) >= self.n:
                efforts = np.asarray(point.effort[: self.n], dtype=float)
            else:
                efforts = np.zeros(self.n, dtype=float)

        point_time = float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9
        return point_time, positions, velocities, efforts

    def trajectory_callback(self, msg: JointTrajectory) -> None:
        if len(msg.points) == 0:
            self.get_logger().warn('Received empty trajectory, ignored.')
            return

        new_traj: List[Tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        for idx in range(len(msg.points)):
            command = self._build_command_from_msg(msg, idx)
            if command is not None:
                new_traj.append(command)

        if not new_traj:
            self.get_logger().warn('Received trajectory without valid positions, ignored.')
            return

        new_traj.sort(key=lambda item: item[0])
        self.trajectory = new_traj
        self.trajectory_start_time = time.monotonic()
        self.position_target = self.trajectory[0][1].copy()
        self.velocity_target = self.trajectory[0][2].copy()
        self.effort_target = self.trajectory[0][3].copy()
        duration = self.trajectory[-1][0]
        self.get_logger().info(
            f"Received {self.command_topic} with {len(self.trajectory)} points over {duration:.3f}s. "
            'Execution restarted from trajectory start time.'
        )

    def sample_trajectory(self, elapsed: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.trajectory:
            return self.default_target, np.zeros(self.n, dtype=float), np.zeros(self.n, dtype=float)

        if elapsed <= self.trajectory[0][0]:
            return self.trajectory[0][1], self.trajectory[0][2], self.trajectory[0][3]

        if elapsed >= self.trajectory[-1][0]:
            return self.trajectory[-1][1], self.trajectory[-1][2], self.trajectory[-1][3]

        for idx in range(1, len(self.trajectory)):
            t1, q1, dq1, tau1 = self.trajectory[idx]
            if elapsed <= t1:
                t0, q0, dq0, tau0 = self.trajectory[idx - 1]
                dt = t1 - t0
                if dt <= 1e-9:
                    return q1, dq1, tau1
                alpha = (elapsed - t0) / dt
                return (
                    (1.0 - alpha) * q0 + alpha * q1,
                    (1.0 - alpha) * dq0 + alpha * dq1,
                    (1.0 - alpha) * tau0 + alpha * tau1,
                )

        return self.trajectory[-1][1], self.trajectory[-1][2], self.trajectory[-1][3]

    def key_callback(self, keycode: int) -> None:
        if keycode >= 0 and chr(keycode) == ' ':
            self.paused = not self.paused

    def _apply_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.config.control_mode == 'direct':
            for ctrl_idx, target in zip(self.ctrl_addrs, self.position_target):
                data.ctrl[ctrl_idx] = float(target)
            return

        if self.config.control_mode == 'impedance_torque':
            q = self._joint_vector_from_data(data, self.qpos_addrs, data.qpos)
            dq = self._joint_vector_from_data(data, self.qvel_addrs, data.qvel)
            tau = (
                self.config.kp * (self.position_target - q)
                + self.config.kd * (self.velocity_target - dq)
            #    + self.effort_target
            )
            tau = np.clip(tau, -self.config.effort_limit, self.config.effort_limit)
            for ctrl_idx, torque in zip(self.ctrl_addrs, tau):
                data.ctrl[ctrl_idx] = float(torque)
            return

        raise ValueError(f"Unsupported control mode '{self.config.control_mode}'")

    def run(self) -> None:
        model = mujoco.MjModel.from_xml_path(self.xml_file)
        data = mujoco.MjData(model)
        self._apply_runtime_model_variations(model, data)
        self._lookup_joint_addresses(model)

        for qpos_idx, target in zip(self.qpos_addrs, self.default_target):
            data.qpos[qpos_idx] = float(target)
        self.position_target = self.default_target.copy()
        self.velocity_target = np.zeros(self.n, dtype=float)
        self.effort_target = np.zeros(self.n, dtype=float)
        self._apply_control(model, data)
        mujoco.mj_forward(model, data)

        with mujoco.viewer.launch_passive(model, data, key_callback=self.key_callback) as viewer:
            while rclpy.ok():
                step_start = time.time()

                if self.trajectory and self.trajectory_start_time is not None:
                    elapsed = time.monotonic() - self.trajectory_start_time
                    self.position_target, self.velocity_target, self.effort_target = self.sample_trajectory(elapsed)
                else:
                    self.position_target = self.default_target
                    self.velocity_target = np.zeros(self.n, dtype=float)
                    self.effort_target = np.zeros(self.n, dtype=float)

                self._apply_control(model, data)

                q = self._joint_vector_from_data(data, self.qpos_addrs, data.qpos)
                dq = self._joint_vector_from_data(data, self.qvel_addrs, data.qvel)

                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.joint_names
                msg.position = [float(x) for x in q]
                msg.velocity = [float(x) for x in dq]
                if data.qfrc_actuator.shape[0] > 0:
                    msg.effort = [float(data.qfrc_actuator[idx]) for idx in self.qvel_addrs]
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
