import os
import pybullet as p
import pybullet_data
import math
import time
import random
from collections import namedtuple
import cv2
import numpy as np
import json
import fpsample

CUBE_LENGTH = 0.05

def visualise_eef_traj(current_eef_pos, prev_line_id=None):
    """
    Visualizes end-effector trajectory by drawing a small sphere at current position
    and a line to the previous position if available.
    
    Args:
        current_eef_pos: Current end-effector position [x, y, z]
        prev_line_id: Previous line ID to maintain trajectory
        
    Returns:
        line_id: ID of the drawn line segment
    """
    # Draw a small red sphere at current EEF position
    p.addUserDebugPoints(
        pointPositions=[current_eef_pos],
        pointColorsRGB=[[1, 0, 0]],  # Red color
        pointSize=5,
        lifeTime=0  # Permanent
    )
    
    # Store previous position for line drawing
    if not hasattr(visualise_eef_traj, 'prev_pos'):
        visualise_eef_traj.prev_pos = None
    
    line_id = None
    if visualise_eef_traj.prev_pos is not None:
        # Draw line from previous to current position
        line_id = p.addUserDebugLine(
            lineFromXYZ=visualise_eef_traj.prev_pos,
            lineToXYZ=current_eef_pos,
            lineColorRGB=[0, 1, 0],  # Green color
            lineWidth=2,
            lifeTime=0  # Permanent
        )
    
    visualise_eef_traj.prev_pos = current_eef_pos
    return line_id

def create_cylinder(radius, height, pos, color=[0.7, 0.7, 0.7, 1]):
    """Creates a cylinder body using primitives."""
    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )
    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )
    body_id = p.createMultiBody(
        baseMass=0,  # Static object
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[pos[0], pos[1], pos[2] + height/2] # Center vertically
    )
    return body_id


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
        self.gripper_range = [-0.04, 0.0] # Prismatic gripper, -0.04=Open, 0.0=Closed
        self.max_velocity = 3

    def load(self):
        self.id = p.loadURDF(
            "./lite-6-updated-urdf/lite_6_new.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()
        self.__print_debug_info__()

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
        p.resetJointState(self.id, self.mimic_parent_id, open_angle)

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            p.resetJointState(self.id, joint_id, open_angle * multiplier)

        for _ in range(10):
            p.stepSimulation()

    def reset_posture(self):
        """Robustly reset the robot to the initial pose and gripper open."""
        # Initial reset posture
        initial_pos_world = [0.125, 0.01, 0.62 + 0.377]
        initial_orn_world = p.getQuaternionFromEuler([3.14, 0, 0])

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


def interpolate_gripper(robot, target_angle,
                        capture_frames=True, iter_folder=None,
                        frame_counter=None, base_pos=None,
                        state_history=None, cube_id=None,
                        cube_pos_history=None, table_id=None,
                        plane_id=None, tray_id=None, EXCLUDE_TABLE=True):
    """Command gripper to target angle until it reaches threshold"""

    # target_angle: input 0.0 (closed) to 1.0 (open) or similar?
    # Original script passed 0.5 (closed-ish) and 0.0 (open).
    
    # Let's map target_angle input from script to Prismatic 
    # Original: 0.0 = Open, 0.8 = Closed (roughly)
    # NEW Prismatic: 0.04 = Open, 0.0 = Closed
    
    # If the calling code passes 0.5 as "Close", we need to map it.
    # calling code 'interpolate_gripper(..., target_angle=0.5)' -> intention: CLOSE?
    # calling code 'interpolate_gripper(..., target_angle=0.0)' -> intention: OPEN?
    
    # Wait, in XArm code:
    # Phase 3: robot.move_gripper(0.5)  -> Close?
    # Phase 6: robot.move_gripper(0.0)  -> Open?
    # let's check XArm move_gripper:
    # "0 = open, 0.8 = closed"
    # So 0.5 is partially closed.
    
    # Start Interpretation:
    # We will assume 'target_angle' passed in is normalized 0(Open) to 0.8(Closed).
    # We need to preserve the logic flow.
    
    # 0.0 (Open) -> Should map to Prismatic -0.04
    # 0.8 (Closed) -> Should map to Prismatic -0.024 (for 5cm box)
    
    if target_angle < 0.01: # asking for Open (0.0)
        prismatic_target = -0.04
    else: # asking for Close (0.5 or 0.8)
        prismatic_target = 0.0 # Forcefully close (target 0.0)
        
    # Override for safety: just use prismatic_target directly if we change calling code?
    # I'll modify the calling code in 'move_and_grab_cube' to be more explicit or use this mapping.
    # For now, let's just use the mapping above to stay compatible with the numerical values in the script tasks.
    
    
    max_iters = 100  # Safety timeout
    for _ in range(max_iters):
        # Control parent joint
        p.setJointMotorControl2(
            robot.id,
            robot.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=prismatic_target,
            force=500,
        )

        # Also control all mimic children joints explicitly
        for joint_id, multiplier in robot.mimic_child_multiplier.items():
            child_target = prismatic_target * multiplier
            p.setJointMotorControl2(
                robot.id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=child_target,
                force=500,
            )

        update_simulation(
            1,
            capture_frames=capture_frames,
            iter_folder=iter_folder,
            frame_counter=frame_counter,
            robot=robot,
            base_pos=base_pos,
            state_history=state_history,
            cube_id=cube_id,
            cube_pos_history=cube_pos_history,
            table_id=table_id,
            plane_id=plane_id,
            tray_id=tray_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        current_gripper_state = p.getJointState(robot.id, robot.mimic_parent_id)
        current_pos = current_gripper_state[0]

        # Stopping condition
        # Closing: Target 0.0, Stop if > -0.024 (Grasped 5cm box)
        # Opening: Target -0.04, Stop if < -0.038
        
        if prismatic_target > -0.01: # Closing (target 0.0)
             if current_pos > -0.024: 
                 break
        else: # Opening (target -0.04)
             if current_pos < -0.038:
                 break

    final_pos = p.getJointState(robot.id, robot.mimic_parent_id)[0]
    print(f"Final gripper position: {final_pos:.4f} (target was {prismatic_target:.4f})")


def create_data_folders(iter_folder):
    tp_rgb_dir = os.path.join(iter_folder, "third_person", "rgb")
    tp_depth_dir = os.path.join(iter_folder, "third_person", "depth")
    tp_pcd_dir = os.path.join(iter_folder, "third_person", "pcd")
    tp_seg_dir = os.path.join(iter_folder, "third_person", "segmentation")

    poses_dir = os.path.join(iter_folder, "camera_poses")

    os.makedirs(tp_rgb_dir, exist_ok=True)
    os.makedirs(tp_depth_dir, exist_ok=True)
    os.makedirs(tp_pcd_dir, exist_ok=True)
    os.makedirs(tp_seg_dir, exist_ok=True)

    os.makedirs(poses_dir, exist_ok=True)

    return {
        "tp_rgb": tp_rgb_dir,
        "tp_depth": tp_depth_dir,
        "tp_pcd": tp_pcd_dir,
        "tp_seg": tp_seg_dir,
        "poses": poses_dir,
    }


def depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix, base_pos, width=224, height=224):
    """Convert depth buffer to 3D point cloud in robot base frame.
    
    Uses PyBullet's own view + projection matrices to unproject
    depth pixels from NDC directly to world space, then to base frame.
    """
    # Pixel coordinates to NDC
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    x_ndc = (2.0 * u / width) - 1.0
    y_ndc = 1.0 - (2.0 * v / height)   # OpenGL: y flips (bottom = -1)
    z_ndc = 2.0 * depth_buffer - 1.0    # depth [0,1] -> NDC [-1,1]

    ndc = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(z_ndc)], axis=-1).reshape(-1, 4)

    # PyBullet returns column-major (OpenGL) -> transpose to row-major
    view_np = np.array(view_matrix).reshape(4, 4).T
    proj_np = np.array(proj_matrix).reshape(4, 4).T

    # clip = Proj @ View @ world_pos  ->  world_pos = inv(Proj @ View) @ clip
    inv_vp = np.linalg.inv(proj_np @ view_np)

    # Unproject NDC to world
    world_homo = (inv_vp @ ndc.T).T
    points_world = world_homo[:, :3] / world_homo[:, 3:4]

    # World -> robot base frame
    points_base = points_world - np.array(base_pos)

    return points_base


def farthest_point_sampling(points, n_samples, colors=None):
    """Farthest Point Sampling using fpsample (Rust-backed).
    
    Args:
        points: (N, 3) array of 3D points
        n_samples: number of points to sample
        colors: optional (N, 3) array of colors
    
    Returns:
        sampled_points: (n_samples, 3) array
        sampled_colors: (n_samples, 3) array if colors provided, else None
    """
    if len(points) <= n_samples:
        return (points, colors) if colors is not None else (points, None)
    
    indices = fpsample.fps_npdu_sampling(points, n_samples)
    sampled_points = points[indices]
    sampled_colors = colors[indices] if colors is not None else None
    return sampled_points, sampled_colors




def save_camera_pose(pose_dict, poses_dir, frame_idx):
    """Save camera pose information"""
    pose_file = os.path.join(poses_dir, f"pose_{frame_idx:04d}.json")
    with open(pose_file, "w") as f:
        json.dump(pose_dict, f, indent=2)


def compute_extrinsics(cam_pos, cam_target, cam_up):
    """Compute camera extrinsics matrix from position, target, and up vector"""
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    cam_up = np.array(cam_up)

    forward = cam_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, cam_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    rotation_matrix = np.array([right, up, -forward])
    translation = cam_pos

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = -rotation_matrix @ translation

    return {
        "rotation_matrix": rotation_matrix.tolist(),
        "translation": translation.tolist(),
        "extrinsics_matrix": extrinsics.tolist(),
    }


def update_simulation(
    steps,
    sleep_time=0.01,
    capture_frames=False,
    iter_folder=None,
    frame_counter=None,
    robot=None,
    base_pos=None,
    state_history=None,
    cube_id=None,
    cube_pos_history=None,
    table_id=None,
    plane_id=None,
    tray_id=None,
    EXCLUDE_TABLE=True,
):
    """Update simulation and capture frames with table segmentation"""

    if capture_frames:
        dirs = create_data_folders(iter_folder)

    # Camera pose from real-world calibration (base-to-camera transform)
    # Translation in base frame: [0.7463, 0.3093, 0.5574]
    # Robot base is at z=0.62 in world frame
    eye = np.array([0.7463, 0.3093, 1.1774])
    direction = np.array([-0.7751, -0.4045, -0.4855])
    direction = direction / np.linalg.norm(direction)
    # tp_cam_eye = [0.7463, 0.3093, 0.62 + 0.5574]  # [0.7463, 0.3093, 1.1774]
    # # Viewing direction from 3rd column of rotation: [-0.7751, -0.4045, -0.4855]
    # tp_cam_target = [0.7463 - 0.7751, 0.3093 - 0.4045, 1.1774 - 0.4855]  # [-0.0288, -0.0952, 0.6919]
    # tp_cam_up = [0, 0, 1]
    tp_cam_eye = (eye - 0.2 * direction).tolist()
    tp_cam_target = (tp_cam_eye + 1.0 * direction).tolist()
    tp_cam_up = [0, 0, 1]

    # Create list of object IDs to exclude from point clouds
    exclude_ids = []
    if table_id is not None and EXCLUDE_TABLE:
        exclude_ids.append(table_id)
    if plane_id is not None:
        exclude_ids.append(plane_id)

    for _ in range(steps):
        p.stepSimulation()

        if capture_frames and iter_folder is not None and frame_counter is not None:
            # ============ THIRD PERSON CAMERA ============
            view_matrix_tp = p.computeViewMatrix(
                cameraEyePosition=tp_cam_eye,
                cameraTargetPosition=tp_cam_target,
                cameraUpVector=tp_cam_up,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
            )

            # Capture with segmentation mask
            width, height, rgb_tp, depth_tp, seg_tp = p.getCameraImage(
                224, 224,
                viewMatrix=view_matrix_tp,
                projectionMatrix=proj_matrix,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            )

            rgb_tp = np.array(rgb_tp)[:, :, :3]
            depth_buffer_tp = np.array(depth_tp)
            seg_tp = np.array(seg_tp)

            # Create mask for objects to exclude
            exclude_mask_tp = np.zeros_like(seg_tp, dtype=bool)
            for obj_id in exclude_ids:
                exclude_mask_tp |= (seg_tp == obj_id)

            # Filter out far-plane (background/sky) pixels
            background_mask = (depth_buffer_tp >= 0.9999).flatten()

            point_cloud_tp = depth_to_point_cloud(
                depth_buffer_tp, view_matrix_tp, proj_matrix, base_pos
            )

            # --- APPLY FILTERS TO NPY DATA ---
            points_tp_flat = point_cloud_tp.reshape(-1, 3)
            valid_mask_tp = points_tp_flat[:, 2] < 2.5
            exclude_mask_flat_tp = exclude_mask_tp.flatten()
            final_mask_tp = valid_mask_tp & (~exclude_mask_flat_tp) & (~background_mask)
            filtered_pcd_tp = points_tp_flat[final_mask_tp]

            # Apply Farthest Point Sampling to get 2500 points
            filtered_pcd_tp, _ = farthest_point_sampling(filtered_pcd_tp, 2500)

            # Save third-person data
            cv2.imwrite(
                os.path.join(dirs["tp_rgb"], f"tp_rgb_{frame_counter[0]:04d}.png"),
                cv2.cvtColor(rgb_tp, cv2.COLOR_RGB2BGR),
            )
            np.save(
                os.path.join(dirs["tp_depth"], f"tp_depth_{frame_counter[0]:04d}.npy"),
                depth_buffer_tp,
            )
            np.save(
                os.path.join(dirs["tp_pcd"], f"tp_pcd_{frame_counter[0]:04d}.npy"),
                filtered_pcd_tp,
            )
            np.save(
                os.path.join(dirs["tp_seg"], f"tp_seg_{frame_counter[0]:04d}.npy"),
                seg_tp,
            )

            # Save PLY with table and background removed + FPS
            colors_tp = rgb_tp.reshape(-1, 3)
            points_tp = point_cloud_tp.reshape(-1, 3)
            exclude_mask_flat_tp = exclude_mask_tp.flatten() | background_mask
            valid_ply = (~exclude_mask_flat_tp) & (points_tp[:, 2] < 2.5)
            ply_points = points_tp[valid_ply]
            ply_colors = colors_tp[valid_ply]

            # Apply FPS to PLY as well
            ply_points, ply_colors = farthest_point_sampling(ply_points, 2500, ply_colors)

            ply_path_tp = os.path.join(
                dirs["tp_pcd"], f"tp_pcd_{frame_counter[0]:04d}.ply"
            )
            save_point_cloud_ply(ply_points, ply_colors, ply_path_tp)

            # ============ SAVE CAMERA POSES ============
            base_pos_array = np.array(base_pos)

            tp_cam_pos_base = np.array(tp_cam_eye) - base_pos_array
            tp_cam_target_base = np.array(tp_cam_target) - base_pos_array
            tp_extrinsics = compute_extrinsics(
                tp_cam_pos_base, tp_cam_target_base, tp_cam_up
            )

            pose_dict = {
                "frame": frame_counter[0],
                "third_person_camera": {
                    "rotation_matrix": tp_extrinsics["rotation_matrix"],
                    "translation": tp_extrinsics["translation"],
                    "extrinsics_matrix": tp_extrinsics["extrinsics_matrix"],
                },
            }

            save_camera_pose(pose_dict, dirs["poses"], frame_counter[0])

            # ============ SAVE ROBOT STATE ============
            current_state = robot.get_robot_state()
            if state_history is not None:
                state_history.append(current_state)

            # ============ SAVE CUBE POSITION ============
            if cube_id is not None and cube_pos_history is not None:
                cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
                cube_state = np.concatenate([np.array(cube_pos), np.array(cube_orn)])
                cube_pos_history.append(cube_state)

            frame_counter[0] += 1


def save_point_cloud_ply(points, colors, filename, exclude_mask=None):
    """Save point cloud in PLY format with colors, excluding masked points"""
    valid_mask = points[:, 2] < 2.5

    if exclude_mask is not None:
        valid_mask = valid_mask & (~exclude_mask)

    points = points[valid_mask]
    colors = colors[valid_mask]

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for point, color in zip(points, colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")


def setup_simulation(freq=240, gui=False):

    if gui:
        p.connect(p.GUI)
        print("PyBullet running in GUI mode")
    else:
        p.connect(p.DIRECT)
        print("PyBullet running in DIRECT (headless) mode")

    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(1/freq)
    plane_id = p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
    table_id = p.loadURDF(
        "table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
    )
    # tray_pos = [0.25, 0.3, 0.6]
    # tray_orn = p.getQuaternionFromEuler([0, 0, 0])
    # # tray_id = p.loadURDF("tray/tray.urdf", tray_pos, tray_orn)
    # tray_id = None

    # return tray_pos, tray_orn, table_id, plane_id, tray_id

    return None, table_id, plane_id


def move_to_pose_dynamic(
    robot, target_pos, target_orn,
    max_steps=200,   # Increased default steps for better settling
    capture_frames=False,
    iter_folder=None,
    frame_counter=None,
    threshold=0.01,  # 1cm accuracy
    **kwargs
):
    """
    Move the robot to a target pose using CLOSED-LOOP control.
    Recalculates IK targets inside the simulation loop for maximum accuracy.
    """
    for step in range(max_steps):
        # 1. Recalculate IK target for current state (Closed-loop)
        robot.move_arm_ik(target_pos, target_orn)
        
        # 2. Advance simulation
        update_simulation(1, robot=robot, capture_frames=capture_frames, 
                          iter_folder=iter_folder, frame_counter=frame_counter, **kwargs)

        # 3. Check if target reached
        current_pos, current_orn = robot.get_current_ee_position()
        # visualise_eef_traj(current_eef_pos=current_pos)
        dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        if dist < threshold:
            # print(f"  Reached target in {step+1} steps.")
            return True

    print(f"  -> Warning: Did not reach target within {max_steps} steps. Final error: {dist:.4f}m")
    return False


def random_color_cube(cube_id):
    color = [random.random(), random.random(), random.random(), 1.0]
    p.changeVisualShape(cube_id, -1, rgbaColor=color)


def move_and_grab_cube(robot, table_id, plane_id, EXCLUDE_TABLE, base_save_dir="dataset"):
    successful_iterations = 0
    total_attempts = 0
    failed_attempts = 0

    print("\n" + "="*60)
    print("STARTING DATA COLLECTION")
    print(f"Target: 150 successful trajectories")
    print("="*60 + "\n")

    # PRIME THE ROBOT: Reset once before the first attempt starts
    # This ensures ATTEMPT 1 starts at the same pose as all others.
    print("Priming robot for collection...")
    robot.reset_posture()

    cube_id = None
    cylinder_id = None
    while successful_iterations < 150:
        # Create temp folder for this attempt
        temp_folder = os.path.join(base_save_dir, f"temp_iter_{total_attempts:04d}")
        os.makedirs(temp_folder, exist_ok=True)

        # Clean up previous cube
        if cube_id is not None:
            try:
                p.removeBody(cube_id)
            except:
                pass
        cube_id = None

        # Clean up previous cylinder
        if cylinder_id is not None:
            try:
                p.removeBody(cylinder_id)
            except:
                pass
        cylinder_id = None

        # Spawn cylinder at a random position on the table for this trajectory
        target_radius = 0.05  # 10cm diameter
        target_height = 0.04  # 4cm height
        # tray_pos = [random.uniform(0.20, 0.35), random.uniform(0.15, 0.25), 0.625]
        const_x_pos = 0.30
        const_y_pos = 0.20
        const_z_pos = 0.625
        tray_pos = [const_x_pos, const_y_pos, const_z_pos]
        cylinder_id = create_cylinder(target_radius, target_height, tray_pos, color=[1, 0.5, 0, 1])
        print(f"Cylinder spawned at: [{tray_pos[0]:.4f}, {tray_pos[1]:.4f}, {tray_pos[2]:.4f}]")

        frame_counter = [0]
        state_history = []
        cube_pos_history = []

        print(f"\n{'='*60}")
        print(f"ATTEMPT {total_attempts + 1} (Successful: {successful_iterations}/150)")
        print(f"{'='*60}\n")

        # Robust Reset: Teleport instantly and reset velocities
        print("Resetting robot posture and gripper...")
        robot.reset_posture()

        actual_pos = p.getJointState(robot.id, robot.mimic_parent_id)[0]
        print(f"Robot reset complete. Gripper pos: {actual_pos:.4f}\n")

        # Spawn random cube
        # cube_start_pos = [random.uniform(0.15, 0.35), random.uniform(-0.2, 0.1), 0.65]
        cube_start_pos = [const_x_pos, -const_y_pos, 0.65]
        print(f"Y distance between cube and cylinder: {abs(cube_start_pos[1] - tray_pos[1]):.4f}m")
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orn)
        # Increase friction of the cube itself
        p.changeDynamics(cube_id, -1, lateralFriction=5.0, spinningFriction=5, rollingFriction=5)
        # Cube color: Black
        p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0, 0, 1])

        print(f"Cube spawned at: [{cube_start_pos[0]:.4f}, {cube_start_pos[1]:.4f}, {cube_start_pos[2]:.4f}]")

        eef_state = robot.get_current_ee_position()
        eef_orientation = eef_state[1]

        # Phase 1: Move above cube
        print("Phase 1: Moving above cube...")
        move_to_pose_dynamic(
            robot, [cube_start_pos[0], cube_start_pos[1], 0.92], eef_orientation,
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Phase 2: Move down to grasp
        print("Phase 2: Moving down to grasp...")
        move_to_pose_dynamic(
            robot, [cube_start_pos[0], cube_start_pos[1], 0.82], eef_orientation,
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Phase 3: Close gripper
        print("Phase 3: Closing gripper...")
        interpolate_gripper(
            robot, target_angle=0.5, # 0.5 passed here usually means "Close" in the old script context
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Phase 4: Lift cube
        print("Phase 4: Lifting cube...")
        move_to_pose_dynamic(
            robot, [cube_start_pos[0], cube_start_pos[1], 0.90], eef_orientation,
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Phase 5: Move above cylinder
        print("Phase 5: Moving above cylinder target...")
        # Target the top of the cylinder [0.3, 0.3, 0.625 + 0.04] + half cube height [0.025] + safety margin
        cylinder_height = 0.04
        target_drop_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + cylinder_height + CUBE_LENGTH/2 + 0.21]
        print(f"  Target position: [{target_drop_pos[0]:.4f}, {target_drop_pos[1]:.4f}, {target_drop_pos[2]:.4f}]")

        move_to_pose_dynamic(
            robot, target_drop_pos, eef_orientation,
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Phase 6: Open gripper to release
        print("Phase 6: Opening gripper to release...")
        interpolate_gripper(
            robot, target_angle=0.0, # 0.0 usually means "Open"
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )

        # Phase 7: Wait for cube to settle
        print("Phase 7: Waiting for cube to settle...")
        for _ in range(500):
            p.stepSimulation()

        # ============ CHECK SUCCESS ============
        # Cylinder Top Success: Cube centered on cylinder and resting on its surface
        cube_final_pos = p.getBasePositionAndOrientation(cube_id)[0]
        dist_to_center = np.linalg.norm(np.array(cube_final_pos[:2]) - np.array(tray_pos[:2]))
        
        cylinder_height = 0.04
        cylinder_radius = 0.05
        cylinder_top_z = tray_pos[2] + cylinder_height
        
        # Success if cube is on cylinder (dist < radius) and high enough (z > cylinder_top)
        # Cube center should be at cylinder_top_z + CUBE_LENGTH/2 = 0.665 + 0.025 = 0.69
        inside_tray = dist_to_center < cylinder_radius and cube_final_pos[2] > cylinder_top_z + 0.01

        cube_final_pos = p.getBasePositionAndOrientation(cube_id)[0]

        if inside_tray:
            print(f"\n{'='*60}")
            print(f"🎉 SUCCESS! Cube on cylinder at {cube_final_pos}")
            print(f"{'='*60}\n")

            # Save this successful trajectory
            final_folder = os.path.join(base_save_dir, f"iter_{successful_iterations:04d}")

            # Rename temp folder to final folder
            if os.path.exists(final_folder):
                import shutil
                shutil.rmtree(final_folder)
            os.rename(temp_folder, final_folder)

            # Save state-action data
            agent_pos = np.array(state_history)

            # Create "Next State" actions (Absolute positions)
            actions = np.zeros_like(agent_pos)
            if len(agent_pos) > 1:
                actions[:-1] = agent_pos[1:]
                actions[-1] = agent_pos[-1]

            # --- FILTERING: Remove static frames ---
            deltas = actions - agent_pos
            dists = np.linalg.norm(deltas, axis=1)

            keep_mask = dists >= 0.008

            if len(keep_mask) > 0:
                keep_mask[-1] = True

            original_count = len(agent_pos)
            agent_pos = agent_pos[keep_mask]
            actions = actions[keep_mask]

            cube_positions = np.array(cube_pos_history)
            if len(cube_positions) == original_count:
                cube_positions = cube_positions[keep_mask]

            print(f"  Filtered static frames: {original_count} -> {len(agent_pos)} (Removed {original_count - len(agent_pos)})")

            # --- PRUNE image/pcd/depth/seg/pose files to match filtered states ---
            removed_indices = set(np.where(~keep_mask)[0])
            if removed_indices:
                prune_dirs = {
                    "tp_rgb":   (os.path.join(final_folder, "third_person", "rgb"),   "tp_rgb_{:04d}.png"),
                    "tp_depth": (os.path.join(final_folder, "third_person", "depth"), "tp_depth_{:04d}.npy"),
                    "tp_pcd":   (os.path.join(final_folder, "third_person", "pcd"),   "tp_pcd_{:04d}.npy"),
                    "tp_ply":   (os.path.join(final_folder, "third_person", "pcd"),   "tp_pcd_{:04d}.ply"),
                    "tp_seg":   (os.path.join(final_folder, "third_person", "segmentation"), "tp_seg_{:04d}.npy"),
                    "cam_pose": (os.path.join(final_folder, "camera_poses"),          "pose_{:04d}.json"),
                }
                for label, (dir_path, pattern) in prune_dirs.items():
                    if not os.path.isdir(dir_path):
                        continue
                    # 1) Delete removed files
                    for idx in removed_indices:
                        fpath = os.path.join(dir_path, pattern.format(idx))
                        if os.path.exists(fpath):
                            os.remove(fpath)
                    # 2) Renumber survivors sequentially
                    ext = pattern.split(".")[-1]
                    prefix = pattern.split("_")[:-1]  # e.g. ["tp", "rgb"]
                    surviving = sorted(f for f in os.listdir(dir_path) if f.endswith(f".{ext}"))
                    for new_idx, old_name in enumerate(surviving):
                        new_name = pattern.format(new_idx)
                        if old_name != new_name:
                            os.rename(
                                os.path.join(dir_path, old_name),
                                os.path.join(dir_path, new_name),
                            )
                print(f"  Pruned data files: kept {len(agent_pos)}/{original_count} frames")


            state_action_data = {
                "agent_pos": agent_pos.tolist(),
                "action": actions.tolist(),
                "cube_pos": cube_positions.tolist(),
                "num_frames": len(agent_pos),
                "state_dim": 7,
                "cube_dim": 7,
                "success": True,
                "attempt_number": total_attempts,
                "success_number": successful_iterations,
                "cube_start_pos": cube_start_pos,
                "cube_final_pos": list(cube_final_pos),
                "cylinder_pos": tray_pos,
                "state_description": {
                    "arm_joints": [0, 1, 2, 3, 4, 5],
                    "gripper": [6],
                },
                "action_description": "Absolute target state (next state) for each timestep",
                "cube_description": {
                    "position": [0, 1, 2],
                    "orientation": [3, 4, 5, 6],
                },
            }

            state_action_file = os.path.join(final_folder, "state_action.json")
            with open(state_action_file, "w") as f:
                json.dump(state_action_data, f, indent=2)

            np.save(os.path.join(final_folder, "agent_pos.npy"), agent_pos)
            np.save(os.path.join(final_folder, "actions.npy"), actions)
            np.save(os.path.join(final_folder, "cube_pos.npy"), cube_positions)

            np.savetxt(
                os.path.join(final_folder, "agent_pos.txt"),
                agent_pos,
                fmt="%.6f",
                delimiter=" ",
                header="6 joint angles + gripper angle normalized (7 dimensions)",
            )
            np.savetxt(
                os.path.join(final_folder, "actions.txt"),
                actions,
                fmt="%.6f",
                delimiter=" ",
                header="Absolute Target States: The state reached at t+1 (7 dimensions)",
            )
            np.savetxt(
                os.path.join(final_folder, "cube_pos.txt"),
                cube_positions,
                fmt="%.6f",
                delimiter=" ",
                header="Cube position (x,y,z) + orientation quaternion (x,y,z,w) (7 dimensions)",
            )

            print(f"Saved successful trajectory to: {final_folder}")
            print(f"  Frames captured: {frame_counter[0]}")
            print(f"  Agent states shape: {agent_pos.shape}")
            print(f"  Actions shape: {actions.shape}")
            print(f"  Cube positions shape: {cube_positions.shape}")

            successful_iterations += 1

        else:
            print(f"\n{'='*60}")
            print(f"❌ FAILED - Cube missed cylinder, ended at {cube_final_pos}")
            print(f"{'='*60}\n")

            # Delete temp folder for failed attempt
            import shutil
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)

            failed_attempts += 1

        # Remove cube and cylinder (will be recreated next iteration)
        p.removeBody(cube_id)
        p.removeBody(cylinder_id)
        cube_id = None
        cylinder_id = None

        total_attempts += 1

        # Print progress
        print(f"\n{'='*60}")
        print(f"PROGRESS UPDATE")
        print(f"{'='*60}")
        print(f"Total attempts: {total_attempts}")
        print(f"Successful: {successful_iterations}/150 ({100*successful_iterations/36:.1f}%)")
        print(f"Failed: {failed_attempts}")
        print(f"Success rate: {100*successful_iterations/total_attempts:.1f}%")
        print(f"Remaining: {150 - successful_iterations}")
        print(f"{'='*60}\n")

    # Final summary
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)
    print(f"Total attempts: {total_attempts}")
    print(f"Successful trajectories saved: {successful_iterations}")
    print(f"Failed attempts (discarded): {failed_attempts}")
    print(f"Overall success rate: {100*successful_iterations/total_attempts:.1f}%")
    print("="*60 + "\n")


def main():

    EXCLUDE_TABLE = True

    """
    If excluding table from point clouds, set EXCLUDE_TABLE = True
    Or else False to include table in point clouds
    """

    _, table_id, plane_id = setup_simulation(freq=60, gui=False)
    robot = Lite6Robot([0, 0, 0.62], [0, 0, 0])
    robot.load()
    move_and_grab_cube(robot, table_id, plane_id, EXCLUDE_TABLE=EXCLUDE_TABLE,base_save_dir="/scratch3/cross-emb/dataset_pick_place_poco")


if __name__ == "__main__":
    main()
