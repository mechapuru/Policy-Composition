import gym
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import pybullet as p
import pybullet_data
import cv2
from termcolor import cprint
from gym import spaces
import random
import time
import os
from collections import namedtuple



def compute_workspace_bounds(pc_xyz, n_std=30):
    """
    Compute workspace bounds using mean ± n_std * std
    Args:
        pc_xyz: (N, 3) numpy array of point cloud coordinates
        n_std: number of standard deviations for bounds
    Returns:
        WORK_SPACE: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """
    if pc_xyz.shape[0] == 0:
        return None

    mean = pc_xyz.mean(axis=0)
    std = pc_xyz.std(axis=0)

    WORK_SPACE = [
        [mean[0] - n_std * std[0], mean[0] + n_std * std[0]],  # X
        [mean[1] - n_std * std[1], mean[1] + n_std * std[1]],  # Y
        [mean[2] - n_std * std[2], mean[2] + n_std * std[2]],  # Z
    ]

    return WORK_SPACE


def crop_workspace(pc_xyz, workspace_bounds):
    """
    Crop point cloud to workspace bounds
    Args:
        pc_xyz: (N, 3) numpy array
        workspace_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    Returns:
        mask: boolean array of valid points
    """
    if workspace_bounds is None:
        return np.ones(pc_xyz.shape[0], dtype=bool)

    mask = (
        (pc_xyz[:, 0] >= workspace_bounds[0][0])
        & (pc_xyz[:, 0] <= workspace_bounds[0][1])
        & (pc_xyz[:, 1] >= workspace_bounds[1][0])
        & (pc_xyz[:, 1] <= workspace_bounds[1][1])
        & (pc_xyz[:, 2] >= workspace_bounds[2][0])
        & (pc_xyz[:, 2] <= workspace_bounds[2][1])
    )

    return mask


def downsample_with_fps(points: np.ndarray, num_points: int = 6000):
    """Fast point cloud sampling using torch3d"""
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points_tensor = torch.tensor([num_points]).cuda()

    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(
        points=points[..., :3], K=num_points_tensor
    )

    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]

    return points


def depth_to_point_cloud(
    depth_buffer,
    view_matrix,
    proj_matrix,
    width=224,
    height=224,
    fov=60,
    near=0.01,
    far=3.0,
    max_depth=2.5,
    exclude_mask=None,
):
    """
    Convert a PyBullet depth buffer to a 3D point cloud in world coordinates,
    optionally filtering points beyond max_depth.

    Args:
        depth_buffer: depth buffer from PyBullet (H x W)
        view_matrix: PyBullet camera view matrix (16-element list)
        proj_matrix: PyBullet projection matrix (16-element list)
        width: image width
        height: image height
        fov: field of view (degrees)
        near: near clipping plane
        far: far clipping plane
        max_depth: maximum distance to keep points (in meters)
    """
    # Convert depth buffer to linear depth
    depth_img = far * near / (far - (far - near) * depth_buffer)

    # Camera intrinsics
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx, cy = width / 2, height / 2

    # Pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # 3D points in camera frame
    z = depth_img
    x = -(u - cx) * z / fx
    y = -(v - cy) * z / fy  # negative to flip y-axis

    points_camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Filter points beyond max_depth
    valid_mask = points_camera[:, 2] < max_depth

    if exclude_mask is not None:  # to exclude table and plane
        valid_mask = valid_mask & (~exclude_mask)

    points_camera = points_camera[valid_mask]
    return points_camera


class UR5Robotiq85:
    """UR5 Robot with Robotiq 85 Gripper"""

    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 3

        # CRITICAL: Gripper normalization limit (same as data collection)
        self.GRIPPER_NORM_LIMIT = 0.2
        self.current_gripper_velocity = 0.0  # Track current velocity command


    def load(self):
        self.id = p.loadURDF(
            "/home/cross-emb/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env/pybullet/urdf/ur5_robotiq_85.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
        from collections import namedtuple

        jointInfo = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
        )
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED

            if controllable:
                self.controllable_joints.append(jointID)

            self.joints.append(
                jointInfo(
                    jointID,
                    jointName,
                    jointType,
                    jointLowerLimit,
                    jointUpperLimit,
                    jointMaxForce,
                    jointMaxVelocity,
                    controllable,
                )
            )

        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][
            : self.arm_num_dofs
        ]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][
            : self.arm_num_dofs
        ]
        self.arm_joint_ranges = [
            ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)
        ]

    def __setup_mimic_joints__(self):
        """Setup mimic joints for Robotiq gripper with STRONG constraints"""
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        
        # Find parent joint ID
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        
        # Store child joint info
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }
        
        # Create constraints with STRONG force (same as data collection)
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            # CRITICAL: High maxForce and erp=1 for stiff constraints
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

            
    def get_robot_state(self, include_gripper=True):
        """
        Get complete robot state
        
        Args:
            include_gripper: If True, return [eef_pos(3), eef_orn_euler(3), joint_angles(6), gripper_normalized(1)]
                           If False, return [eef_pos(3), eef_orn_euler(3), joint_angles(6)]
        
        Returns: 
            state array of size 13 (with gripper) or 12 (without gripper)
        """
        eef_state = p.getLinkState(self.id, self.eef_id)
        eef_pos = np.array(eef_state[0])
        eef_orn_quat = np.array(eef_state[1])
        eef_orn_euler = np.array(p.getEulerFromQuaternion(eef_orn_quat))

        joint_states = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)
            joint_states.append(joint_state[0])

        if include_gripper:
            # Get raw gripper angle and normalize
            raw_angle = p.getJointState(self.id, self.mimic_parent_id)[0]
            
            # Noise suppression
            if abs(raw_angle) < 1e-3:
                raw_angle = 0.0
            
            # Normalize to [0, 1] using GRIPPER_NORM_LIMIT
            gripper_normalized = np.clip(raw_angle / self.GRIPPER_NORM_LIMIT, 0.0, 1.0)
            state = np.concatenate([eef_pos, eef_orn_euler, joint_states, [gripper_normalized]])
        else:
            state = np.concatenate([eef_pos, eef_orn_euler, joint_states])

        return state
    

    def get_joint_positions(self, include_gripper=True):
        """
        Get current joint positions
        
        Args:
            include_gripper: If True, return [arm_joints(6), gripper_normalized(1)]
                           If False, return [arm_joints(6)]
        
        Returns:
            joint positions array
        """
        joint_positions = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)
            joint_positions.append(joint_state[0])

        if include_gripper:
            # Get normalized gripper value
            raw_angle = p.getJointState(self.id, self.mimic_parent_id)[0]
            if abs(raw_angle) < 1e-3:
                raw_angle = 0.0
            gripper_normalized = np.clip(raw_angle / self.GRIPPER_NORM_LIMIT, 0.0, 1.0)
            joint_positions.append(gripper_normalized)

        return np.array(joint_positions)


    def get_joint_limits(self, include_gripper=True):
        """
        Get joint limits
        
        Args:
            include_gripper: If True, include gripper limits
        
        Returns:
            limits_lower, limits_upper
        """
        limits_lower = self.arm_lower_limits
        limits_upper = self.arm_upper_limits
        
        if include_gripper:
            limits_lower = limits_lower + [self.gripper_range[0]]
            limits_upper = limits_upper + [self.gripper_range[1]]
            
        return np.array(limits_lower), np.array(limits_upper)

    def set_arm_joints(self, joint_positions):
        """Set arm joint positions directly"""
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                joint_positions[i],
                maxVelocity=self.max_velocity,
            )

    def move_gripper(self, velocity_command):
        """
        Velocity-based gripper control (same as data collection)
        
        Args:
            velocity_command: Discrete control input
                -1.0: Open gripper (executed as -0.25 for safety)
                 0.0: Hold current state
                 1.0: Close gripper
        """
        # Store the velocity command
        self.current_gripper_velocity = velocity_command
        
        # Map discrete command to actual velocity
        if velocity_command == -1.0:
            actual_velocity = -0.25 * 20.0  # Open slowly
        elif velocity_command == 0.0:
            actual_velocity = 0.0  # Hold
        elif velocity_command == 1.0:
            actual_velocity = 1.0 * 20.0  # Close
        else:
            # Continuous value - scale it
            cprint("Warning: Using continuous gripper velocity command.", "red")
            actual_velocity = velocity_command * 20.0
        
        p.setJointMotorControl2(
            self.id, 
            self.mimic_parent_id, 
            p.VELOCITY_CONTROL,
            targetVelocity=actual_velocity,
            force=100  # High force for reliable movement
        )

            
    def set_joint_positions(self, joint_positions, include_gripper=True):
        """
        Set all joint positions
        
        Args:
            joint_positions: If include_gripper=True: [arm_joints(6), gripper_velocity_cmd(1)]
                           If include_gripper=False: [arm_joints(6)]
            include_gripper: Whether gripper command is included
        """
        # Set arm joints
        self.set_arm_joints(joint_positions[:6])
        
        # Set gripper using velocity command (if included)
        if include_gripper and len(joint_positions) > 6:
            gripper_cmd = joint_positions[6]
            self.move_gripper(gripper_cmd)


    def get_eef_position(self):
        """Get end-effector 3D position"""
        eef_state = p.getLinkState(self.id, self.eef_id)
        return np.array(eef_state[0])


class UR5PickPlaceEnv(gym.Env):
    """
    PyBullet UR5 Pick and Place Environment
    
    Supports both 6D (arm only) and 7D (arm + gripper) action spaces
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        use_gui=False,
        num_points=6000,
        image_size=224,
        use_workspace_crop=True,
        workspace_std=30.0,
        action_dim=7,
        capture_table=False,
    ):
        self.use_gui = use_gui
        self.num_points = num_points
        self.image_size = image_size
        self.max_steps = 350
        self.current_step = 0

        self.use_workspace_crop = use_workspace_crop
        self.workspace_std = workspace_std
        self.workspace_bounds = None

        self.action_dim = action_dim
        self.capture_table = capture_table
        
        # NEW: Determine if gripper is included
        self.include_gripper = (action_dim == 7 or action_dim == 13)
        
        cprint(f"Environment initialized with action_dim={action_dim}", "cyan")
        cprint(f"Include gripper: {self.include_gripper}", "cyan")

        # Connect to PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.table_id = p.loadURDF(
            "table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
        )

        self.tray_pos = [0.5, 0.9, 0.6]
        self.tray_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.tray_id = p.loadURDF("tray/tray.urdf", self.tray_pos, self.tray_orn)

        # Load robot
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # Camera parameters
        self.tp_cam_eye = [1.1, -0.6, 1.3]
        self.tp_cam_target = [0.5, 0.3, 0.7]
        self.tp_cam_up = [0, 0, 1]

        self.cube_id = None
        self.cube_start_pos = None

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(self.action_dim,), dtype=np.float32
        )

        self.observation_space = spaces.Dict(
            {
                "point_cloud": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_points, 3),
                    dtype=np.float32,
                ),
                "agent_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32
                ),
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self.image_size, self.image_size),
                    dtype=np.uint8,
                ),
            }
        )

        self.is_success_flag = False


    def reset(self, cube_start_pos=None, cube_start_orn=None):
        """
        Reset the environment

        Args:
            cube_start_pos: Optional 3D position [x, y, z] for cube. If None, random position is used.
            cube_start_orn: Optional quaternion [x, y, z, w] for cube orientation. If None, [0, 0, 0, 1] is used.
        """
        cprint(f"IN ENV RESET parameters {cube_start_pos} and {cube_start_orn}", "cyan")

        self.current_step = 0
        self.is_success_flag = False
        self.workspace_bounds = None

        # Remove old cube if exists
        if self.cube_id is not None:
            p.removeBody(self.cube_id)

        # Move to rest pose
        rest_pose = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            p.setJointMotorControl2(
                self.robot.id, joint_id, p.POSITION_CONTROL, rest_pose[i]
            )

        if self.include_gripper:
            p.setJointMotorControl2(
                self.robot.id, 
                self.robot.mimic_parent_id, 
                p.POSITION_CONTROL, 
                targetPosition=0.000,
                force=1500,
                maxVelocity=1.5
            )
    
        for _ in range(5000):
            p.stepSimulation()
        
        # Stabilize
        print("\nStabilizing robot at rest pose...")
        for _ in range(1000):
            p.stepSimulation()

        if self.include_gripper:
            actual_angle = p.getJointState(self.robot.id, self.robot.mimic_parent_id)[0]
            print("Reset complete. Gripper position after reset: {:.4f}\n".format(actual_angle))

        # Use provided cube position or generate random one
        if cube_start_pos is not None:
            self.cube_start_pos = cube_start_pos
        else:
            cprint("Randomizing cube start position.", "red")
            self.cube_start_pos = [
                random.uniform(0.3, 0.7),
                random.uniform(-0.1, 0.1),
                0.65,
            ]

        # Use provided orientation or default to identity quaternion
        if cube_start_orn is not None:
            if len(cube_start_orn) == 4:
                cube_orn_quat = cube_start_orn
            else:
                cube_orn_quat = p.getQuaternionFromEuler(cube_start_orn)
        else:
            cprint("Using default cube(random) orientation.", "red")
            cube_orn_quat = p.getQuaternionFromEuler([0, 0, 0])

        self.cube_id = p.loadURDF("cube_small.urdf", self.cube_start_pos, cube_orn_quat)

        # Random cube color
        color = [random.random(), random.random(), random.random(), 1.0]
        p.changeVisualShape(self.cube_id, -1, rgbaColor=color)

        # Step simulation
        for _ in range(50):
            p.stepSimulation()

        obs = self._get_obs()
        return obs

    def step(self, action):
        """
        Execute action and return observation.
        
        Args:
            action: Action array. Can be:
                - 6D: [arm_deltas(6)] - arm only, no gripper
                - 7D: [arm_deltas(6), gripper_velocity_cmd(1)]
                - 13D: [eef_delta(6), arm_deltas(6), gripper_velocity_cmd(1)]
        """
        self.current_step += 1

        # Handle different action dimensions
        if len(action) == 13:
            joint_deltas = action[6:13]
            arm_deltas = joint_deltas[:6]
            if self.include_gripper:
                gripper_velocity_cmd = joint_deltas[6]
        elif len(action) == 7:
            arm_deltas = action[:6]
            if self.include_gripper:
                gripper_velocity_cmd = action[6]
        elif len(action) == 6:
            # 6D action - arm only, no gripper
            arm_deltas = action[:6]
            gripper_velocity_cmd = None
        else:
            raise ValueError(f"Action must be 6D, 7D, or 13D, got {len(action)}D")

        # Discretize gripper command if included
        if self.include_gripper and gripper_velocity_cmd is not None:
            if gripper_velocity_cmd < -0.33:
                gripper_cmd_discrete = -1.0
            elif gripper_velocity_cmd > 0.33:
                gripper_cmd_discrete = 1.0
            else:
                gripper_cmd_discrete = 0.0
        else:
            gripper_cmd_discrete = None

        # Get current state
        current_joint_pos = self.robot.get_joint_positions(include_gripper=self.include_gripper)

        # Update arm joints
        target_arm = current_joint_pos[:6] + arm_deltas
        self.robot.set_arm_joints(target_arm)

        # Update gripper if included
        if self.include_gripper and gripper_cmd_discrete is not None:
            self.robot.move_gripper(gripper_cmd_discrete)

        for _ in range(1000):
            p.stepSimulation()

        time.sleep(0.55)

        obs = self._get_obs()

        # Check success: cube in tray
        reward = 0.0
        done = False
        if self.cube_id is not None:
            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            if (
                abs(cube_pos[0] - self.tray_pos[0]) < 0.2
                and abs(cube_pos[1] - self.tray_pos[1]) < 0.2
                and abs(cube_pos[2] - self.tray_pos[2]) < 0.1
            ):
                reward = 1.0
                self.is_success_flag = True
                done = True

        # Episode timeout
        if self.current_step >= self.max_steps:
            done = True

        info = {"is_success": self.is_success_flag}
        return obs, reward, done, info

    def _get_obs(self):
        """Get current observation"""
        robot_state = self.robot.get_robot_state(include_gripper=self.include_gripper)

        # Extract appropriate agent_pos based on action_dim
        if self.action_dim == 6:
            # 6D: [eef_pos(3), eef_orn_euler(3), joint_angles(6)] -> use last 6 (joint angles)
            agent_pos = robot_state[6:12]
        elif self.action_dim == 7:
            # 7D: [joint_angles(6), gripper(1)]
            agent_pos = robot_state[6:13]
        elif self.action_dim == 13:
            # 13D: full state
            agent_pos = robot_state
        else:
            raise ValueError(
                f"Unsupported action_dim: {self.action_dim}. Must be 6, 7, or 13."
            )

        # Exclude table from point cloud
        exclude_ids = []
        if self.table_id is not None and not self.capture_table:
            exclude_ids.append(self.table_id)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.tp_cam_eye,
            cameraTargetPosition=self.tp_cam_target,
            cameraUpVector=self.tp_cam_up,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
        )

        # Capture with segmentation
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            self.image_size,
            self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )

        rgb_img = np.array(rgb_img)[:, :, :3]
        depth_buffer = np.array(depth_img)
        seg_img = np.array(seg_img)

        exclude_mask_tp = np.zeros_like(seg_img, dtype=bool)
        for obj_id in exclude_ids:
            exclude_mask_tp |= seg_img == obj_id

        exclude_mask_flat_tp = exclude_mask_tp.flatten()

        point_cloud = depth_to_point_cloud(
            depth_buffer,
            view_matrix,
            proj_matrix,
            self.image_size,
            self.image_size,
            exclude_mask=exclude_mask_flat_tp,
        )

        # Workspace cropping
        if self.use_workspace_crop:
            if self.workspace_bounds is None:
                self.workspace_bounds = compute_workspace_bounds(
                    point_cloud, n_std=self.workspace_std
                )
                if self.workspace_bounds is not None:
                    print(
                        f"Computed workspace bounds (mean ± {self.workspace_std}*std):"
                    )
                    print(
                        f"  X: [{self.workspace_bounds[0][0]:.3f}, {self.workspace_bounds[0][1]:.3f}]"
                    )
                    print(
                        f"  Y: [{self.workspace_bounds[1][0]:.3f}, {self.workspace_bounds[1][1]:.3f}]"
                    )
                    print(
                        f"  Z: [{self.workspace_bounds[2][0]:.3f}, {self.workspace_bounds[2][1]:.3f}]"
                    )

            if self.workspace_bounds is not None:
                mask = crop_workspace(point_cloud, self.workspace_bounds)
                point_cloud = point_cloud[mask]
                if self.current_step == 0:
                    print(f"  After workspace crop: {point_cloud.shape[0]} points")

        # Downsample or pad to target number of points
        if point_cloud.shape[0] == 0:
            point_cloud = np.zeros((self.num_points, 3), dtype=np.float32)
        elif point_cloud.shape[0] > self.num_points:
            point_cloud = downsample_with_fps(point_cloud, self.num_points)
        elif point_cloud.shape[0] < self.num_points:
            padding = np.zeros((self.num_points - point_cloud.shape[0], 3))
            point_cloud = np.vstack([point_cloud, padding])

        rgb_img = rgb_img.transpose(2, 0, 1)

        obs_dict = {
            "point_cloud": point_cloud.astype(np.float32),
            "agent_pos": agent_pos.astype(np.float32),
            "image": rgb_img.astype(np.uint8),
        }

        return obs_dict

    def render(self, mode="rgb_array"):
        """Render the environment"""
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.tp_cam_eye,
            cameraTargetPosition=self.tp_cam_target,
            cameraUpVector=self.tp_cam_up,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
        )

        width, height, rgb_img, _, _ = p.getCameraImage(
            self.image_size,
            self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
        )

        rgb_img = np.array(rgb_img)[:, :, :3]
        return rgb_img.astype(np.uint8)

    def close(self):
        """Close the environment"""
        p.disconnect(self.physics_client)

    def is_success(self):
        """Check if task was successful"""
        return self.is_success_flag

    def seed(self, seed=None):
        """Set random seed"""
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)


_LITE6_MP_CAMERAS = [
    {
        "name": "thirdperson_cam00",
        "eye": [0.6294, 0.9766, 1.3748],
        "target": [0.20, 0.00, 0.75],
        "up": [0.0, 0.0, 1.0],
    },
    {
        "name": "thirdperson_cam01",
        "eye": [0.8562, -0.5951, 1.4672],
        "target": [0.20, 0.00, 0.75],
        "up": [0.0, 0.0, 1.0],
    },
    {
        "name": "thirdperson_cam02",
        "eye": [-0.0628, 0.0000, 1.9990],
        "target": [0.25, 0.00, 0.70],
        "up": [0.0, 1.0, 0.0],
    },
]


def _depth_to_point_cloud_base_frame(
    depth_buffer,
    view_matrix,
    proj_matrix,
    base_pos,
    width=224,
    height=224,
):
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    x_ndc = (2.0 * u / width) - 1.0
    y_ndc = 1.0 - (2.0 * v / height)
    z_ndc = 2.0 * depth_buffer - 1.0
    ndc = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(z_ndc)], axis=-1).reshape(-1, 4)

    view_np = np.array(view_matrix).reshape(4, 4).T
    proj_np = np.array(proj_matrix).reshape(4, 4).T
    inv_vp = np.linalg.inv(proj_np @ view_np)

    world_homo = (inv_vp @ ndc.T).T
    points_world = world_homo[:, :3] / world_homo[:, 3:4]
    points_base = points_world - np.array(base_pos, dtype=np.float32)
    return points_base


def _normalize_point_count(points: np.ndarray, num_points: int):
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if points.shape[0] > num_points:
        return downsample_with_fps(points.astype(np.float32, copy=False), num_points)
    if points.shape[0] < num_points:
        repeat_n = num_points // points.shape[0]
        rem = num_points % points.shape[0]
        tiled = np.tile(points, (repeat_n, 1))
        if rem > 0:
            tiled = np.concatenate([tiled, points[:rem]], axis=0)
        return tiled.astype(np.float32, copy=False)
    return points.astype(np.float32, copy=False)


def create_static_box(position, half_extents, yaw=0.0, color=[0.8, 0.3, 0.3, 1]):
    orientation_quat = p.getQuaternionFromEuler([0, 0, yaw])
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation_quat,
    )


def create_static_sphere(position, radius, color=[0.2, 0.9, 0.2, 1.0]):
    visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
    )


def generate_task_obstacles(start_pos, goal_pos, obs_config):
    obstacles = []
    p_start = np.array(start_pos, dtype=np.float32)
    p_goal = np.array(goal_pos, dtype=np.float32)
    midpoint = p_start + 0.5 * (p_goal - p_start)

    obs1_size_cfg = obs_config.get("obs1_size", [0.02, 0.15, 0.1])
    max_y = obs1_size_cfg[1]
    max_z = obs1_size_cfg[2]
    size_1 = [
        random.uniform(0.01, 0.03),
        random.uniform(0.05, max_y - 0.1),
        random.uniform(0.05, max_z),
    ]
    yaw_1 = random.uniform(-1.57 / 2, 1.57 / 2)
    obstacles.append(
        create_static_box(position=midpoint.tolist(), half_extents=size_1, yaw=yaw_1, color=[0.8, 0.2, 0.2, 1.0])
    )

    trap_strategy = random.choice(["horizontal_wall", "plate", "ignore"])
    if trap_strategy == "ignore":
        return obstacles

    if trap_strategy == "horizontal_wall":
        trap_pos = midpoint + np.array([random.choice([-0.1, 0.1]), 0.0, -0.05], dtype=np.float32)
        yaw_2 = 0.0
        size_2 = [0.02, 0.25, 0.04]
        color_2 = [0.8, 0.8, 0.2, 1.0]
    else:
        y_shift = random.uniform(-0.05, 0.05)
        trap_pos = midpoint + np.array([0.05, y_shift, 0.0], dtype=np.float32)
        trap_pos[2] = max(float(start_pos[2]), float(goal_pos[2])) + 0.15
        yaw_2 = 0.0
        size_2 = [0.1, 0.05, 0.1]
        color_2 = [0.2, 0.8, 0.8, 1.0]

    obstacles.append(
        create_static_box(position=trap_pos.tolist(), half_extents=size_2, yaw=yaw_2, color=color_2)
    )
    return obstacles

class Lite6Robot:
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        # link_eef index: in lite_6_new.urdf, link_eef is index 6 (joint_eef)
        # joint indices: 0..5 are arm joints, 6 is joint_eef
        self.eef_id = 6
        self.arm_num_dofs = 6
        # Lite6 rest poses - keeping similar to XArm for now, can be tuned
        self.arm_rest_poses = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0] 
        self.gripper_range = [-0.04, -0.028] # Prismatic gripper, -0.04=Open, -0.028=Grasp
        self.max_velocity = 3
        self.urdf_path = "/home/cross-emb/nitin_exp/dp3_motion/3D-Diffusion-Policy/diffusion_policy_3d/env/pybullet/lite-6-updated-urdf/lite_6_new.urdf"

    def load(self):
        self.id = p.loadURDF(
            self.urdf_path,
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()
        # self.__print_debug_info__()

    def __parse_joint_info__(self):
        jointInfo = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
        )
        self.joints = []
        self.controllable_joints = []
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            # print(f"Joint {i}: {info[1].decode('utf-8')}, Type: {info[2]}, Limits: [{info[8]}, {info[9]}], MaxForce: {info[10]}, MaxVelocity: {info[11]}")
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(
                    jointID,
                    jointName,
                    jointType,
                    jointLowerLimit,
                    jointUpperLimit,
                    jointMaxForce,
                    jointMaxVelocity,
                    controllable,
                )
            )
        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][
            : self.arm_num_dofs
        ]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][
            : self.arm_num_dofs
        ]
        # Manually overriding base joint limits for better IK performance (can be tuned)
        self.arm_lower_limits[0] = -1.7
        self.arm_upper_limits[0] = 1.7

        self.arm_joint_ranges = [
            ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)
        ]

    def __setup_mimic_joints__(self):
        """Setup mimic joints for Lite6 custom gripper"""
        # Parent joint is the one we control directly
        mimic_parent_name = "left_finger_joint"
        # Children follow the parent
        mimic_children_names = {
            "right_finger_joint": 1, 
        }

        # Find parent joint ID
        try:
            self.mimic_parent_id = [
                joint.id for joint in self.joints if joint.name == mimic_parent_name
            ][0]
        except IndexError:
            print(f"Error: Could not find mimic parent joint '{mimic_parent_name}'")
            # Fallback or exit? For now let it crash to debug
            raise

        # Store child joint info
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }

        # Create constraints with strong force
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0], # Axis usually doesn't matter for GEAR, ratio does
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            # Prismatic to Prismatic gear ratio 1:1 usually works directly
            # For GEAR constraint: jointB = jointA * gearRatio
            # Here right_finger = left_finger * 1
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100000, erp=1)

        # Increase friction for gripper pads to improve grasp
        gripper_links = [self.mimic_parent_id] + list(self.mimic_child_multiplier.keys())
        for link_id in gripper_links:
            p.changeDynamics(
                self.id,
                link_id,
                lateralFriction=5.0,
                spinningFriction=5.0,
                rollingFriction=5.0,
            )

    def __print_debug_info__(self):
        """Print debug info about joint mapping and eef position"""
        print("\n" + "="*60)
        print("ROBOT DEBUG INFO")
        print("="*60)
        jtype_names = {0: 'REVOLUTE', 1: 'PRISMATIC', 4: 'FIXED'}
        for j in self.joints:
            jtype = jtype_names.get(j.type, str(j.type))
            ctrl = "CTRL" if j.controllable else "    "
            print(f"  Joint {j.id:2d}: {j.name:<35s} ({jtype:<9s}) [{ctrl}]")
        print(f"\nArm controllable joints: {self.arm_controllable_joints}")
        print(f"Arm lower limits: {self.arm_lower_limits}")
        print(f"Arm upper limits: {self.arm_upper_limits}")
        print(f"EEF link index: {self.eef_id}")
        eef_state = p.getLinkState(self.id, self.eef_id)
        print(f"EEF position: {eef_state[0]}")
        print(f"EEF orientation (quat): {eef_state[1]}")
        print(f"Mimic parent (finger_joint) ID: {self.mimic_parent_id}")
        print(f"Mimic children: {self.mimic_child_multiplier}")
        print("="*60 + "\n")

    def move_arm_ik(self, target_pos, target_orn):
        joint_poses = p.calculateInverseKinematics(
            self.id,
            self.eef_id,
            target_pos,
            target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                joint_poses[i],
                maxVelocity=self.max_velocity,
            )

    def move_gripper(self, open_angle):
        """
        Move gripper to target angle.
        
        Args:
            open_angle: Target position for left_finger_joint (meters)
            open_angle: Target position for left_finger_joint (meters)
                        0.0 = Closed (fingers touch)
                        -0.04 = Open (fingers apart)
                        -0.024 = Grasp (5cm box)
        """
        prismatic_target = float(np.clip(open_angle, self.gripper_range[0], self.gripper_range[1]))

        max_iters = 100
        for _ in range(max_iters):
            p.setJointMotorControl2(
                self.id,
                self.mimic_parent_id,
                p.POSITION_CONTROL,
                targetPosition=prismatic_target,
                force=500,
            )

            for joint_id, multiplier in self.mimic_child_multiplier.items():
                child_target = prismatic_target * multiplier
                p.setJointMotorControl2(
                    self.id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=child_target,
                    force=500,
                )

            p.stepSimulation()

            current_pos = p.getJointState(self.id, self.mimic_parent_id)[0]
            if abs(current_pos - prismatic_target) < 2e-3:
                break

    def reset_posture(self):
        """Robustly reset the robot to the initial pose and gripper open."""
        # Initial reset posture
        initial_pos_world = [0.125, 0.01, 0.62 + 0.377]
        # Old orientation not considering the tcp axes
        # initial_orn_world = p.getQuaternionFromEuler([3.14, 0, 0])
        # Now considering the TCP orientation as well, which is rotated 90 degrees around Y from the base link
        initial_orn_world = p.getQuaternionFromEuler([-1.5708,0, 1.5708])

        # Reset joints to rest_poses FIRST
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, self.arm_rest_poses[i], targetVelocity=0)

        target_joint_positions = p.calculateInverseKinematics(
            self.id,
            self.eef_id,
            initial_pos_world,
            initial_orn_world,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )

        # Disable motor control during teleport to avoid fighting
        for j in range(p.getNumJoints(self.id)):
            p.setJointMotorControl2(self.id, j, p.VELOCITY_CONTROL, force=0)

        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, target_joint_positions[i], targetVelocity=0)
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, targetPosition=target_joint_positions[i], force=1000)

        # Reset gripper joints to open (-0.04 for prismatic)
        # NOTE: -val is Open
        gripper_open_val = -0.04
        p.resetJointState(self.id, self.mimic_parent_id, gripper_open_val, targetVelocity=0)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=gripper_open_val, force=500)
        
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            p.resetJointState(self.id, joint_id, gripper_open_val * multiplier, targetVelocity=0)
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, targetPosition=gripper_open_val * multiplier, force=500)

        # Settle physics
        for _ in range(100):
            p.stepSimulation()

    def get_current_ee_position(self):
        eef_state = p.getLinkState(self.id, self.eef_id)
        return eef_state[0], eef_state[1]

    def get_robot_state(self):
        """Get robot state: 6 arm joints + 1 gripper normalized [0, 1]"""
        joint_states = [p.getJointState(self.id, i)[0] for i in self.arm_controllable_joints]

        # Get raw gripper angle (position in meters)
        raw_gripper_pos = p.getJointState(self.id, self.mimic_parent_id)[0]

        # User request: Map [Open, Grasp] to [0, 1].
        # Open = -0.04 -> 0.0
        # Grasp = -0.024 -> 1.0
        # Anything tighter than -0.024 is capped at 1.0
        
        open_pos = -0.04
        grasp_pos = -0.028 # Target close position for 5cm box
        
        if abs(grasp_pos - open_pos) < 1e-5:
            normalized_gripper = 0.0
        else:
            normalized_gripper = (raw_gripper_pos - open_pos) / (grasp_pos - open_pos)
        
        normalized_gripper = min(max(normalized_gripper, 0.0), 1.0)

        # Return only joint angles + gripper (7 dimensions)
        state = np.concatenate([joint_states, [normalized_gripper]])
        return state


class Lite6MotionPlanningEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        use_gui=False,
        num_points=5000,
        image_size=224,
        action_dim=7,
        max_steps=350,
        collision_terminate=False,
        collision_distance_threshold=0.01,
        success_threshold=0.03,
        obs_config=None,
        max_steps_per_waypoint=10,
        waypoint_threshold=0.01,
    ):
        self.use_gui = use_gui
        self.num_points = num_points
        self.image_size = image_size
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.collision_terminate = collision_terminate
        self.collision_distance_threshold = max(0.0, float(collision_distance_threshold))
        self.success_threshold = success_threshold
        self.obs_config = obs_config or {}
        self.max_steps_per_waypoint = max_steps_per_waypoint
        self.waypoint_threshold = waypoint_threshold

        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))

        self.robot = Lite6Robot([0, 0, 0.62], [0, 0, 0])
        self.robot.load()
        self.goal_marker_id = None
        for i in range(p.getNumJoints(self.robot.id)):
            joint_info = p.getJointInfo(self.robot.id, i)
            # index 1 is the joint name, index 12 is the child link name
            link_name = joint_info[12].decode('utf-8') 
            if link_name == "tcp":
                tcp_link_index = i
                print(f"Found TCP link '{link_name}' at index {tcp_link_index}")
                break

        if tcp_link_index != -1:
            self.robot.eef_id = tcp_link_index
        

        self.goal_eef_xyz = np.array([0.3, 0.0, 0.8], dtype=np.float32)
        self.start_eef_xyz = np.array([0.25, -0.2, 0.75], dtype=np.float32)
        self.target_orn = p.getQuaternionFromEuler([-1.5708, 0, 1.5708])
        self.show_goal_marker = True
        self.goal_marker_radius = 0.015
        self.goal_marker_color = [0.1, 0.9, 0.1, 1.0]
        self.obstacle_ids = []
        self.collision_flag = False
        self.is_success_flag = False
        self.last_cam_frames = {}

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "point_cloud": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_points, 3), dtype=np.float32),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                "goal_eef_xyz": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "image": spaces.Box(low=0, high=255, shape=(3, self.image_size, self.image_size), dtype=np.uint8),
            }
        )

    def _capture_camera(self, cam_cfg):
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_cfg["eye"],
            cameraTargetPosition=cam_cfg["target"],
            cameraUpVector=cam_cfg["up"],
        )
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=3.0)
        _, _, rgb_img, depth_img, seg_img = p.getCameraImage(
            self.image_size,
            self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )
        rgb = np.asarray(rgb_img)[:, :, :3]
        depth = np.asarray(depth_img)
        seg = np.asarray(seg_img)
        return rgb, depth, seg, view_matrix, proj_matrix

    def _build_point_cloud(self):
        merged = []
        exclude_ids = {self.table_id, self.plane_id}
        if self.goal_marker_id is not None:
            exclude_ids.add(self.goal_marker_id)
        for cam_cfg in _LITE6_MP_CAMERAS:
            rgb, depth, seg, view_matrix, proj_matrix = self._capture_camera(cam_cfg)
            self.last_cam_frames[cam_cfg["name"]] = rgb

            points = _depth_to_point_cloud_base_frame(
                depth,
                view_matrix,
                proj_matrix,
                self.robot.base_pos,
                width=self.image_size,
                height=self.image_size,
            )
            points_flat = points.reshape(-1, 3)
            seg_flat = seg.flatten()
            depth_flat = depth.flatten()

            valid = (depth_flat < 0.9999) & (points_flat[:, 2] < 2.5)
            for obj_id in exclude_ids:
                valid &= (seg_flat != obj_id)
            filtered = points_flat[valid]
            if filtered.size > 0:
                merged.append(filtered)

        if len(merged) == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)

        points = np.concatenate(merged, axis=0).astype(np.float32, copy=False)
        return _normalize_point_count(points, self.num_points)

    def _check_collision(self):
        for body_id in [self.table_id] + list(self.obstacle_ids):
            if self.collision_distance_threshold <= 0.0:
                contacts = p.getContactPoints(self.robot.id, body_id)
                for contact in contacts:
                    robot_link = contact[3]
                    if robot_link == -1:
                        continue
                    return True
            else:
                closest_points = p.getClosestPoints(
                    self.robot.id,
                    body_id,
                    distance=self.collision_distance_threshold,
                )
                for point in closest_points:
                    robot_link = point[3]
                    if robot_link == -1:
                        continue
                    # point[8] is distance in meters; it can be negative for penetration.
                    if point[8] <= self.collision_distance_threshold:
                        return True
        return False

    def _get_obs(self):
        pc = self._build_point_cloud()
        state = self.robot.get_robot_state()
        cam0 = self.last_cam_frames.get("thirdperson_cam00", np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        return {
            "point_cloud": pc.astype(np.float32),
            "agent_pos": state.astype(np.float32),
            "goal_eef_xyz": self.goal_eef_xyz.astype(np.float32),
            "image": cam0.transpose(2, 0, 1).astype(np.uint8),
        }

    def _dataset_gripper_to_joint(self, gripper_cmd):
        """Map dataset gripper command [0, 1] to Lite6 finger joint [-0.04, 0.0]."""
        cmd = float(np.clip(gripper_cmd, 0.0, 1.0))
        open_joint = self.robot.gripper_range[0]   # -0.04 (open)
        close_joint = self.robot.gripper_range[1]  # 0.0 (closed)
        return open_joint + cmd * (close_joint - open_joint)

    def _clear_goal_marker(self):
        if self.goal_marker_id is None:
            return
        try:
            p.removeBody(self.goal_marker_id)
        except Exception:
            pass
        self.goal_marker_id = None

    def _update_goal_marker(self):
        self._clear_goal_marker()
        if not self.show_goal_marker:
            return
        self.goal_marker_id = create_static_sphere(
            position=self.goal_eef_xyz.astype(np.float32).tolist(),
            radius=self.goal_marker_radius,
            color=self.goal_marker_color,
        )

    def reset(self, start_joint=None, goal_eef_xyz=None, cube_start_pos=None, cube_start_orn=None):
        del start_joint, cube_start_pos, cube_start_orn
        self.current_step = 0
        self.collision_flag = False
        self.is_success_flag = False

        for obs_id in self.obstacle_ids:
            try:
                p.removeBody(obs_id)
            except Exception:
                pass
        self.obstacle_ids = []

        self.robot.reset_posture()

        self.start_eef_xyz = np.array([0.25, -0.2, 0.75], dtype=np.float32)

        self.goal_eef_xyz = np.array([
            0.35 + np.random.uniform(-0.05, 0.0),
            0.20 + np.random.uniform(-0.05, 0.05),
            0.75 + float(np.random.uniform(-0.05, 0.2, size=1)[0]),
        ], dtype=np.float32)

        self._update_goal_marker()

        self.obstacle_ids = generate_task_obstacles(
            self.start_eef_xyz.tolist(),
            self.goal_eef_xyz.tolist(),
            self.obs_config,
        )

        start_joint_target = p.calculateInverseKinematics(
            self.robot.id,
            self.robot.eef_id,
            self.start_eef_xyz.tolist(),
            self.target_orn,
            lowerLimits=self.robot.arm_lower_limits,
            upperLimits=self.robot.arm_upper_limits,
            jointRanges=self.robot.arm_joint_ranges,
            restPoses=self.robot.arm_rest_poses,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            p.resetJointState(self.robot.id, joint_id, start_joint_target[i], targetVelocity=0)
            p.setJointMotorControl2(
                self.robot.id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=start_joint_target[i],
                force=1000,
            )
        # Dataset convention: 0=open, 1=close.
        self.robot.move_gripper(self._dataset_gripper_to_joint(0.0))
        for _ in range(120):
            p.stepSimulation()

        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Lite6MotionPlanningEnv expects 7D action, got {action.shape}")

        target_arm = action[:6]
        gripper_cmd_dataset = float(action[6])
        current_gripper_pos = self.robot.get_robot_state()[-1]  # Last element is gripper state [0, 1]
        gripper_target_joint = self._dataset_gripper_to_joint(current_gripper_pos+gripper_cmd_dataset)
        self.robot.move_gripper(gripper_target_joint)

        collided = False
        for _ in range(self.max_steps_per_waypoint):
            for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                p.setJointMotorControl2(
                    self.robot.id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=float(target_arm[i]),
                    force=150.0,
                    maxVelocity=self.robot.max_velocity,
                    positionGain=0.05,
                    velocityGain=1.0,
                )

            for _ in range(1):
                p.stepSimulation()

            if self._check_collision():
                collided = True
                if self.collision_terminate:
                    break

            current_joint_pos = np.array(
                [p.getJointState(self.robot.id, j)[0] for j in self.robot.arm_controllable_joints],
                dtype=np.float32,
            )
            if np.max(np.abs(current_joint_pos - target_arm)) < self.waypoint_threshold:
                break

        self.collision_flag = self.collision_flag or collided

        eef_pos = self.robot.get_current_ee_position()[0]
        goal_dist = float(np.linalg.norm(eef_pos - self.goal_eef_xyz))
        reached = goal_dist <= self.success_threshold
        self.is_success_flag = reached and (not self.collision_flag)

        reward = 1.0 if self.is_success_flag else 0.0
        done = self.is_success_flag or (self.current_step >= self.max_steps)
        if self.collision_terminate and collided:
            done = True

        obs = self._get_obs()
        info = {
            "is_success": self.is_success_flag,
            "collision": collided,
            "collision_ever": self.collision_flag,
            "goal_dist": goal_dist,
        }
        return obs, reward, done, info

    def render_camera(self, camera_index=0):
        camera_index = int(np.clip(camera_index, 0, len(_LITE6_MP_CAMERAS) - 1))
        cam_cfg = _LITE6_MP_CAMERAS[camera_index]
        rgb, _, _, _, _ = self._capture_camera(cam_cfg)
        return rgb.astype(np.uint8)

    def render(self, mode="rgb_array"):
        del mode
        return self.render_camera(0)

    def close(self):
        self._clear_goal_marker()
        p.disconnect(self.physics_client)

    def is_success(self):
        return self.is_success_flag

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)


class Lite6PickPlaceEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        use_gui=False,
        num_points=2500,
        image_size=224,
        action_dim=7,
        max_steps=350,
        obs_config=None,
        max_steps_per_waypoint=4,
        waypoint_threshold=0.02,
    ):
        self.use_gui = use_gui
        self.num_points = num_points
        self.image_size = image_size
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.obs_config = obs_config or {}
        self.max_steps_per_waypoint = max_steps_per_waypoint
        self.waypoint_threshold = waypoint_threshold

        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1/240)

        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))

        self.robot = Lite6Robot([0, 0, 0.62], [0, 0, 0])
        self.robot.load()
        
        self.cube_id = None
        self.tray_id = None
        self.is_success_flag = False
        self.last_cam_frames = {}

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "point_cloud": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_points, 3), dtype=np.float32),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                "image": spaces.Box(low=0, high=255, shape=(3, self.image_size, self.image_size), dtype=np.uint8),
            }
        )

    def _capture_camera(self):
        # Using the same fixed camera as data generation
        eye = np.array([0.7463, 0.3093, 1.1774])
        direction = np.array([-0.7751, -0.4045, -0.4855])
        direction = direction / np.linalg.norm(direction)
        cam_eye = (eye - 0.2 * direction).tolist()
        cam_target = (cam_eye + 1.0 * direction).tolist()
        cam_up = [0, 0, 1]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=cam_target,
            cameraUpVector=cam_up,
        )
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=3.0)
        _, _, rgb_img, depth_img, seg_img = p.getCameraImage(
            self.image_size,
            self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )
        rgb = np.asarray(rgb_img)[:, :, :3]
        depth = np.asarray(depth_img)
        seg = np.asarray(seg_img)
        return rgb, depth, seg, view_matrix, proj_matrix

    def _build_point_cloud(self):
        rgb, depth, seg, view_matrix, proj_matrix = self._capture_camera()
        self.last_cam_frames["main"] = rgb

        points = _depth_to_point_cloud_base_frame(
            depth, view_matrix, proj_matrix, self.robot.base_pos,
            width=self.image_size, height=self.image_size,
        )
        points_flat = points.reshape(-1, 3)
        seg_flat = seg.flatten()
        depth_flat = depth.flatten()

        valid = (depth_flat < 0.9999) & (points_flat[:, 2] < 2.5)
        exclude_ids = {self.table_id, self.plane_id}
        for obj_id in exclude_ids:
            valid &= (seg_flat != obj_id)
        
        filtered = points_flat[valid]
        if filtered.size == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)
        
        return _normalize_point_count(filtered, self.num_points)

    def reset(self, cube_start_pos=None, cube_start_orn=None):
        self.current_step = 0
        self.is_success_flag = False

        if self.cube_id is not None:
             p.removeBody(self.cube_id)
        if self.tray_id is not None:
             p.removeBody(self.tray_id)

        self.robot.reset_posture()

        # Load tray (cylinder from data script)
        # Position from data script: [0.1, 0.05, 0.70]
        tray_pos = [0.1, 0.05, 0.62] 
        self.tray_id = p.loadURDF("tray/tray.urdf", tray_pos, p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.5)

        # Load cube
        if cube_start_pos is None:
            # Fixed start position from data collection script
            cube_start_pos = [0.3, -0.2, 0.65]
        if cube_start_orn is None:
            cube_start_orn = [0, 0, 0, 1]
            
        self.cube_id = p.loadURDF("cube/cube.urdf", cube_start_pos, cube_start_orn, globalScaling=1.0)
        p.changeDynamics(self.cube_id, -1, mass=0.1, lateralFriction=1.0)

        for _ in range(50):
            p.stepSimulation()

        return self._get_obs()

    def _get_obs(self):
        pc = self._build_point_cloud()
        state = self.robot.get_robot_state()
        cam0 = self.last_cam_frames.get("main", np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
        return {
            "point_cloud": pc.astype(np.float32),
            "agent_pos": state.astype(np.float32),
            "image": cam0.transpose(2, 0, 1).astype(np.uint8),
        }

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        
        target_arm = action[:6]
        gripper_cmd_dataset = float(action[6])
        
        # Similar to MP step but simplified for PP
        for _ in range(self.max_steps_per_waypoint):
            for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                p.setJointMotorControl2(
                    self.robot.id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=float(target_arm[i]),
                    force=150.0,
                    maxVelocity=self.robot.max_velocity,
                )
            
            # Gripper control
            current_gripper = self.robot.get_robot_state()[-1]
            target_gripper = current_gripper + gripper_cmd_dataset
            self.robot.move_gripper(self.robot._dataset_gripper_to_joint(target_gripper))

            p.stepSimulation()

        # Success check: cube in tray
        reward = 0.0
        done = False
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        tray_pos, _ = p.getBasePositionAndOrientation(self.tray_id)
        dist = np.linalg.norm(np.array(cube_pos[:2]) - np.array(tray_pos[:2]))
        if dist < 0.1 and abs(cube_pos[2] - tray_pos[2]) < 0.1:
            reward = 1.0
            self.is_success_flag = True
            done = True

        if self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {"is_success": self.is_success_flag}

    def is_success(self):
        return self.is_success_flag

    def get_video(self):
        return self.render_camera()
    
    def render_camera(self):
        rgb, _, _, _, _ = self._capture_camera()
        return rgb.astype(np.uint8)

    def render(self, mode="rgb_array"):
        return self.render_camera()
    
    def close(self):
        p.disconnect(self.physics_client)

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)