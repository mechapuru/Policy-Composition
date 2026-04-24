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

import ompl.base as ob 
import ompl.geometric as og
from planner import RobotOMPLPlanner, solve_ik_collision_free,solve_ik, visualise_eef_traj 
import test_lite6_ompl
import shutil as _shutil
from utils import interactive_camera_helper

CUBE_LENGTH = 0.05
import yaml

def generate_task_obstacles_old(point_A_pos, point_B_pos, num_obs):
    midpoint = [(point_A_pos[0]+point_B_pos[0])/2, (point_A_pos[1]+point_B_pos[1])/2, 0.62]
    wall_extents = [0.05, 0.02, 0.15] 
    wall_id = p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_extents),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=wall_extents, rgbaColor=[0.8, 0.2, 0.2, 1]),
        basePosition=[midpoint[0], midpoint[1], midpoint[2] + wall_extents[2]] 
    )
    obstacle_ids = []
    x_min,xmax = min(point_A_pos[0], point_B_pos[0]), max(point_A_pos[0], point_B_pos[0])
    y_min, y_max = min(point_A_pos[1], point_B_pos[1]), max(point_A_pos[1], point_B_pos[1])
    z_min = min(point_A_pos[2], point_B_pos[2])

    # A. The Primary Wall (Manual check: ensure Point A/B are at Y=0.2/-0.2 and wall is at Y=0)
    # ... spawn wall_id as before ...
    obstacle_ids.append(wall_id)

    # B. Additional Constrained & Filtered Obstacles
    safety_distance = 0.08  # 8cm buffer to keep points clear of obstacle centers
    for i in range(num_obs - 1):
        valid_spawn = False
        attempts = 0
        while not valid_spawn and attempts < 100:
            rand_x = random.uniform(x_min, xmax)
            rand_y = random.uniform(y_min, y_max)
            rand_z = random.uniform(z_min - 0.05, z_min + 0.1)
            candidate_pos = [rand_x, rand_y, rand_z]
            
            # Check distance to A and B
            dist_A = np.linalg.norm(np.array(candidate_pos[:2]) - np.array(point_A_pos[:2]))
            dist_B = np.linalg.norm(np.array(candidate_pos[:2]) - np.array(point_B_pos[:2]))
            
            if dist_A > safety_distance and dist_B > safety_distance:
                valid_spawn = True
            attempts += 1

        if valid_spawn:
            pillar_extents = [0.03, 0.03, random.uniform(0.1, 0.2)]
            pid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=pillar_extents),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=pillar_extents, rgbaColor=[0.4, 0.4, 0.4, 1]),
                basePosition=[rand_x, rand_y, rand_z + pillar_extents[2]]
            )
            obstacle_ids.append(pid)
    return obstacle_ids

def create_static_box(position, half_extents, yaw=0.0, color=[0.8, 0.3, 0.3, 1]):
    """Helper to spawn a static box with a specific yaw (Z-axis rotation)"""
    # Convert yaw (rotation in XY plane) to PyBullet quaternion
    orientation_quat = p.getQuaternionFromEuler([0, 0, yaw])
    
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    
    body_id = p.createMultiBody(
        baseMass=0, # Mass 0 means it floats and cannot be pushed
        baseCollisionShapeIndex=col_id, 
        baseVisualShapeIndex=vis_id, 
        basePosition=position,
        baseOrientation=orientation_quat
    )
    return body_id

def generate_task_obstacles(start_pos, goal_pos, obs_config):
    """
    Generates Obstacle 1 (Dynamic Center Blocker) and Obstacle 2 (Plate/Wall/Ignore)
    """
    obstacles = []
    p_start = np.array(start_pos)
    p_goal = np.array(goal_pos)
    path_vector = p_goal - p_start
    # ==========================================
    # OBSTACLE 1: The Configurable Center Blocker
    # ==========================================
    # Locked exactly to the midpoint
    midpoint = p_start + 0.5 * path_vector
    # Read the maximum allowed sizes from config
    max_y = obs_config.get('obs1_size', [0.02, 0.15, 0.1])[1]
    max_z = obs_config.get('obs1_size', [0.02, 0.15, 0.1])[2]
    # Randomly scale it between a small 5cm cube and the max rectangular cuboid
    size_1 = [
        random.uniform(0.01, 0.03),
        random.uniform(0.05, max_y-0.1),
        random.uniform(0.05, max_z) ]
    # Randomize the yaw
    yaw_1 = random.uniform(-1.57/2, 1.57/2) 
    # Fully opaque red
    obs1_id = create_static_box(position=midpoint, half_extents=size_1, yaw=yaw_1, color=[0.8, 0.2, 0.2, 1.0])
    obstacles.append(obs1_id)
    # OBSTACLE 2:
    trap_strategy = random.choice(["horizontal_wall", "plate", "ignore"])
    
    if trap_strategy == "ignore":
        print("Obstacle 2 ignored this run. Only center blocker spawned.")
        return obstacles
        
    elif trap_strategy == "horizontal_wall":
        # Sits on the Y-axis (yaw=0), long in Y, very short in Z.
        # Placed slightly off-center in X so it doesn't perfectly overlap Obstacle 1
        trap_pos = midpoint + np.array([random.choice([-0.1, 0.1]), 0, -0.05])
        yaw_2 = 0.0 # Fixed to Y-axis
        size_2 = [0.02, 0.25, 0.04] # Long, thin, and short
        color_2 = [0.8, 0.8, 0.2, 1.0] # Solid Yellow
    elif trap_strategy == "plate":
        # Flat plate at exact midpoint X and Z, but shifted slightly in Y.
        y_shift = random.uniform(-0.05, 0.05)
        trap_pos = midpoint + np.array([0.05, y_shift, 0])
        trap_pos[2] = max(start_pos[2], goal_pos[2]) + 0.15  # Place it slightly below the EEF height for better blocking
        yaw_2 = 0.0 # Yaw doesn't change
        size_2 = [0.1, 0.05, 0.1] # Flat 1cm thick plate
        color_2 = [0.2, 0.8, 0.8, 1.0] # Solid Cyan
    print(f"Spawning Obstacle 2 using strategy: {trap_strategy}")
    obs2_id = create_static_box(position=trap_pos, half_extents=size_2, yaw=yaw_2, color=color_2)
    obstacles.append(obs2_id) 
    return obstacles
def generate_simple_obstacle(start_pos,goal_pos):
    """Generate a tall tower at the middle of start and end sitting on the table"""
    obstacles = []
    p_start = np.array(start_pos)
    p_goal = np.array(goal_pos)
    height_start = p_start[2] -0.625 # Table height
    height_goal = p_goal[2] -0.625
    midpoint = p_start + 0.5 * (p_goal - p_start)
    # We want to adjust the z position so that the box sits on the table (0.625) and also that total z height > height where eef moves horizontally(0.28m)
    # Additionally it should be obstacle , hence half_extent in z > max_z/2
    box_half_extents = [np.random.uniform(0.02, 0.03), np.random.uniform(0.02, 0.03), np.random.uniform(max(height_start, height_goal)/2, 0.4)] # 10cm x 10cm base, 40cm tall
    box_position = [midpoint[0], midpoint[1], 0.625 + box_half_extents[2]] # Centered at midpoint, but z adjusted to sit on table
    yaw = random.uniform(-1.57/15, 1.57/15) # Random yaw for variety
    box_id = create_static_box(position=box_position, half_extents=box_half_extents, yaw=yaw, color=[0.8, 0.2, 0.2, 1.0])
    obstacles.append(box_id)
    return obstacles



def draw_sphere(position, color=[1,0,0], radius=0.03):
    visual = p.createVisualShape(shapeType=p.GEOM_SPHERE,radius=radius,rgbaColor=color + [1])
    p.createMultiBody(baseMass=0,baseVisualShapeIndex=visual,basePosition=position)

def create_cylinder(radius, height, pos, color=[0.7, 0.7, 0.7, 1]):
    """Creates a cylinder body using primitives."""
    visual_shape = p.createVisualShape(p.GEOM_CYLINDER,radius=radius,length=height,rgbaColor=color)
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER,radius=radius,height=height)
    body_id = p.createMultiBody(baseMass=0,CollisionShapeIndex=collision_shape,baseVisualShapeIndex=visual_shape,basePosition=[pos[0], pos[1], pos[2] + height/2])
    return body_id

def random_color_cube(cube_id):
    color = [random.random(), random.random(), random.random(), 1.0]
    p.changeVisualShape(cube_id, -1, rgbaColor=color)


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
        self.urdf_path = "./lite-6-updated-urdf/lite_6_new.urdf"

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
        p.resetJointState(self.id, self.mimic_parent_id, open_angle)

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            p.resetJointState(self.id, joint_id, open_angle * multiplier)

        for _ in range(10):
            p.stepSimulation()

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


def create_data_folders_3cam(iter_folder):
    """Create folder structure for 3-camera data collection (motion planning)."""
    dirs = {}
    for cam_name in ["thirdperson_cam00", "thirdperson_cam01", "thirdperson_cam02"]:
        for modality in ["rgb", "depth", "pcd", "segmentation"]:
            key = f"{cam_name}_{modality}"
            path = os.path.join(iter_folder, cam_name, modality)
            os.makedirs(path, exist_ok=True)
            dirs[key] = path

    poses_dir = os.path.join(iter_folder, "camera_poses")
    os.makedirs(poses_dir, exist_ok=True)
    dirs["poses"] = poses_dir

    return dirs


# ---------------------------------------------------------------------------
# Three camera definitions for motion_plan_data
# ---------------------------------------------------------------------------
# All positions in world frame.  Robot base is at z=0.62.
# cam00 = Left  tilted cross view (robot seen from left-front)
# cam01 = Right tilted cross view (robot seen from right-front)
# cam02 = Top   view (directly above)
_MP_CAMERAS = [
    {
        "name": "thirdperson_cam00",
        "eye":    [0.6294, 0.9766, 1.3748],
        "target": [ 0.20,  0.00, 0.75],
        "up":     [ 0.0,   0.0,  1.0 ],
    },
    {
        "name": "thirdperson_cam01",
        "eye":    [0.8562, -0.5951, 1.4672],
        "target": [ 0.20,  0.00, 0.75],
        "up":     [ 0.0,   0.0,  1.0 ],
    },
    {
        "name": "thirdperson_cam02",
        "eye":    [-0.0628, 0.0000, 1.9990],
        "target": [ 0.25,  0.00, 0.70],
        "up":     [ 0.0,   1.0,  0.0 ],
    },
]


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


def _capture_single_camera(cam_cfg, base_pos, exclude_ids, dirs, cam_name, frame_idx):
    """
    Capture one camera view and save rgb / depth / pcd (npy + ply) / segmentation.

    Parameters
    ----------
    cam_cfg    : dict  – keys: name, eye, target, up
    base_pos   : list  – robot base position in world frame
    exclude_ids: list  – body IDs to strip from point clouds
    dirs       : dict  – output from create_data_folders_3cam()
    cam_name   : str   – e.g. "thirdperson_cam00"
    frame_idx  : int   – current frame index
    """
    eye    = cam_cfg["eye"]
    target = cam_cfg["target"]
    up     = cam_cfg["up"]

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=up,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
    )

    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        224, 224,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    )

    rgb_arr   = np.array(rgb_img)[:, :, :3]
    depth_arr = np.array(depth_img)
    seg_arr   = np.array(seg_img)

    # Build exclusion & background masks
    exclude_mask = np.zeros_like(seg_arr, dtype=bool)
    for obj_id in exclude_ids:
        exclude_mask |= (seg_arr == obj_id)
    background_mask = (depth_arr >= 0.9999).flatten()

    # Unproject to point cloud in robot base frame
    pcd_world = depth_to_point_cloud(depth_arr, view_matrix, proj_matrix, base_pos)
    pts_flat  = pcd_world.reshape(-1, 3)

    # Filter
    valid_z    = pts_flat[:, 2] < 2.5
    excl_flat  = exclude_mask.flatten()
    final_mask = valid_z & (~excl_flat) & (~background_mask)
    filtered_pts = pts_flat[final_mask]
    filtered_pts, _ = farthest_point_sampling(filtered_pts, 2500)

    fid = f"{frame_idx:04d}"

    # Save RGB
    cv2.imwrite(
        os.path.join(dirs[f"{cam_name}_rgb"],  f"{cam_name}_rgb_{fid}.png"),
        cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR),
    )
    # Save depth
    np.save(os.path.join(dirs[f"{cam_name}_depth"], f"{cam_name}_depth_{fid}.npy"), depth_arr)
    # Save PCD (npy)
    np.save(os.path.join(dirs[f"{cam_name}_pcd"],   f"{cam_name}_pcd_{fid}.npy"),   filtered_pts)
    # Save segmentation
    np.save(os.path.join(dirs[f"{cam_name}_segmentation"], f"{cam_name}_seg_{fid}.npy"), seg_arr)

    # Save PLY (coloured)
    cols_flat = rgb_arr.reshape(-1, 3)
    ply_excl  = excl_flat | background_mask
    ply_valid = (~ply_excl) & (pts_flat[:, 2] < 2.5)
    ply_pts   = pts_flat[ply_valid]
    ply_cols  = cols_flat[ply_valid]
    ply_pts, ply_cols = farthest_point_sampling(ply_pts, 2500, ply_cols)
    save_point_cloud_ply(
        ply_pts, ply_cols,
        os.path.join(dirs[f"{cam_name}_pcd"], f"{cam_name}_pcd_{fid}.ply"),
    )

    # Return extrinsics for pose dict
    base_arr = np.array(base_pos)
    ext = compute_extrinsics(
        np.array(eye)    - base_arr,
        np.array(target) - base_arr,
        up,
    )
    return {
        "eye_world":    eye,
        "target_world": target,
        "up":           up,
        "rotation_matrix":  ext["rotation_matrix"],
        "translation":      ext["translation"],
        "extrinsics_matrix": ext["extrinsics_matrix"],
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
    use_3cam=False,
):
    """Update simulation and capture frames.

    When *use_3cam* is True (motion-planning mode) three cameras are used:
        thirdperson_cam00  – left  tilted cross view
        thirdperson_cam01  – right tilted cross view
        thirdperson_cam02  – top view
    Otherwise the original single calibrated camera is used (grab-cube mode).
    """

    if capture_frames:
        if use_3cam:
            dirs = create_data_folders_3cam(iter_folder)
        else:
            dirs = create_data_folders(iter_folder)

    # ---- single-camera parameters (grab-cube mode) ----
    eye = np.array([0.7463, 0.3093, 1.1774])
    direction = np.array([-0.7751, -0.4045, -0.4855])
    direction = direction / np.linalg.norm(direction)

    # tp_cam_eye    = [0.7463, 0.3093, 0.62 + 0.5574]
    # tp_cam_target = [0.7463 - 0.7751, 0.3093 - 0.4045, 1.1774 - 0.4855]
    # tp_cam_up     = [0, 0, 1]
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

            if use_3cam:
                # ============ THREE-CAMERA CAPTURE ============
                pose_dict = {"frame": frame_counter[0]}
                for cam_cfg in _MP_CAMERAS:
                    cam_name = cam_cfg["name"]
                    cam_pose = _capture_single_camera(
                        cam_cfg, base_pos, exclude_ids, dirs, cam_name, frame_counter[0]
                    )
                    pose_dict[cam_name] = cam_pose
                save_camera_pose(pose_dict, dirs["poses"], frame_counter[0])

            else:
                # ============ ORIGINAL SINGLE CAMERA ============
                view_matrix_tp = p.computeViewMatrix(
                    cameraEyePosition=tp_cam_eye,
                    cameraTargetPosition=tp_cam_target,
                    cameraUpVector=tp_cam_up,
                )
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
                )

                width, height, rgb_tp, depth_tp, seg_tp = p.getCameraImage(
                    224, 224,
                    viewMatrix=view_matrix_tp,
                    projectionMatrix=proj_matrix,
                    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                )

                rgb_tp         = np.array(rgb_tp)[:, :, :3]
                depth_buffer_tp = np.array(depth_tp)
                seg_tp         = np.array(seg_tp)

                exclude_mask_tp = np.zeros_like(seg_tp, dtype=bool)
                for obj_id in exclude_ids:
                    exclude_mask_tp |= (seg_tp == obj_id)
                background_mask = (depth_buffer_tp >= 0.9999).flatten()

                point_cloud_tp = depth_to_point_cloud(
                    depth_buffer_tp, view_matrix_tp, proj_matrix, base_pos
                )

                points_tp_flat   = point_cloud_tp.reshape(-1, 3)
                valid_mask_tp    = points_tp_flat[:, 2] < 2.5
                exclude_mask_flat_tp = exclude_mask_tp.flatten()
                final_mask_tp    = valid_mask_tp & (~exclude_mask_flat_tp) & (~background_mask)
                filtered_pcd_tp  = points_tp_flat[final_mask_tp]
                filtered_pcd_tp, _ = farthest_point_sampling(filtered_pcd_tp, 2500)

                cv2.imwrite(
                    os.path.join(dirs["tp_rgb"], f"tp_rgb_{frame_counter[0]:04d}.png"),
                    cv2.cvtColor(rgb_tp, cv2.COLOR_RGB2BGR),
                )
                np.save(os.path.join(dirs["tp_depth"], f"tp_depth_{frame_counter[0]:04d}.npy"), depth_buffer_tp)
                np.save(os.path.join(dirs["tp_pcd"],   f"tp_pcd_{frame_counter[0]:04d}.npy"),   filtered_pcd_tp)
                np.save(os.path.join(dirs["tp_seg"],   f"tp_seg_{frame_counter[0]:04d}.npy"),   seg_tp)

                colors_tp = rgb_tp.reshape(-1, 3)
                points_tp = point_cloud_tp.reshape(-1, 3)
                exclude_mask_flat_tp = exclude_mask_tp.flatten() | background_mask
                valid_ply  = (~exclude_mask_flat_tp) & (points_tp[:, 2] < 2.5)
                ply_points = points_tp[valid_ply]
                ply_colors = colors_tp[valid_ply]
                ply_points, ply_colors = farthest_point_sampling(ply_points, 2500, ply_colors)
                save_point_cloud_ply(
                    ply_points, ply_colors,
                    os.path.join(dirs["tp_pcd"], f"tp_pcd_{frame_counter[0]:04d}.ply"),
                )

                base_pos_array    = np.array(base_pos)
                tp_cam_pos_base   = np.array(tp_cam_eye)    - base_pos_array
                tp_cam_target_base = np.array(tp_cam_target) - base_pos_array
                tp_extrinsics     = compute_extrinsics(tp_cam_pos_base, tp_cam_target_base, tp_cam_up)

                pose_dict = {
                    "frame": frame_counter[0],
                    "third_person_camera": {
                        "rotation_matrix":   tp_extrinsics["rotation_matrix"],
                        "translation":       tp_extrinsics["translation"],
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
        dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        if dist < threshold:
            # print(f"  Reached target in {step+1} steps.")
            return True

    print(f"  -> Warning: Did not reach target within {max_steps} steps. Final error: {dist:.4f}m")
    return False


def move_with_planner(planner,robot, target_pos, target_orn,
                      max_steps_per_waypoint=10,capture_frames=False, iter_folder=None,
                      frame_counter=None,threshold=0.01, **kwargs):
    start_pos, start_orn = robot.get_current_ee_position()
    start_config = robot.get_robot_state()[:robot.arm_num_dofs]
    goal_config = solve_ik(robot, planner, target_pos, target_orn)  # Get only arm joints for planning
    if goal_config is None:
        print("✗ Planning failed, IK could not find a solution.")
        return False
    planner._snapshot_gripper_pose()
    success,path = planner.plan(start_config, goal_config,planning_time=10.0)
    if not path:
        print("✗ Planning failed, No path found.")
        return False

    # Visualize planned path
    test_lite6_ompl.visualize_path(robot, path)
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.resetJointState(robot.id, joint_id, start_config[i])
    planner._apply_frozen_gripper_printing()  # Ensure gripper stays in place during execution
    #######################################################
    ### EXECUTE WITH PLANNER (REPLANNING + CLOSED-LOOP) ###
    print(f"\nExecuting path with {len(path)} waypoints...")
    for i, waypoint in enumerate(path):
            # Set joint targets for only joints of lite6 , not the gripper
            for j, joint_id in enumerate(planner.joint_ids):
                p.setJointMotorControl2(
                    planner.robot.id,
                    joint_id,
                    p.POSITION_CONTROL,
                    waypoint[j],
                    maxVelocity=getattr(planner.robot, 'max_velocity', 3.0),
                    force=200
                )
             # Get current end-effector position
            eef_state = p.getLinkState(planner.robot.id, planner.robot.eef_id)
            current_eef_pos = eef_state[0]  # World position of the link

            # Visualize trajectory  
            visualise_eef_traj(current_eef_pos)
                
            # Simulate motion to waypoint
            for _ in range(max_steps_per_waypoint):
                # 2. Advance simulation
                planner._apply_frozen_gripper()  # Ensure gripper stays in place during execution
                update_simulation(1, robot=robot, capture_frames=capture_frames, 
                          iter_folder=iter_folder, frame_counter=frame_counter, **kwargs)
                print(f"Gripper pos: {p.getJointState(robot.id, robot.mimic_parent_id)[0]:.4f}")
                # p.stepSimulation()
                # time.sleep(dt)
                
                # Early exit if close enough
                current = [p.getJointState(planner.robot.id, j)[0] for j in planner.joint_ids]
                if np.max(np.abs(np.array(current) - np.array(waypoint))) < 0.01:
                    break
            
            if i % max(1, len(path) // 10) == 0:
                print(f"  Progress: {i+1}/{len(path)} waypoints")
            # 3. Check if target reached
            current_pos, current_orn = robot.get_current_ee_position()
            dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            if dist < threshold:
                # print(f"  Reached target in {step+1} steps.")
                continue
    return True

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
    while successful_iterations < 5:
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
        tray_pos = [random.uniform(0.20, 0.35), random.uniform(0.15, 0.25), 0.625]
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
        cube_start_pos = [random.uniform(0.15, 0.35), random.uniform(-0.2, 0.1), 0.65]
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orn)
        # Increase friction of the cube itself
        p.changeDynamics(cube_id, -1, lateralFriction=5.0, spinningFriction=5, rollingFriction=5)
        # Cube color: Black
        p.changeVisualShape(cube_id, -1, rgbaColor=[0, 0, 0, 1])

        print(f"Cube spawned at: [{cube_start_pos[0]:.4f}, {cube_start_pos[1]:.4f}, {cube_start_pos[2]:.4f}]")

        eef_state = robot.get_current_ee_position()
        eef_orientation = eef_state[1]
        planner1 = RobotOMPLPlanner(
        robot=robot,
        robot_urdf="./lite-6-updated-urdf/lite_6_new.urdf",
        obstacles=[table_id,cube_id],
        collision_margin=0.02,
        ignore_base=True
        )
        planner1.set_planner("AITStar")

        planner2 = RobotOMPLPlanner(
        robot=robot,
        robot_urdf="./lite-6-updated-urdf/lite_6_new.urdf",
        obstacles=[table_id,cylinder_id],
        collision_margin=0.02,
        ignore_base=True
        )
        planner2.set_planner("AITStar")

        # Phase 1: Move above cube
        print("Phase 1: Moving above cube...")
        # move_to_pose_dynamic(
        #     robot, [cube_start_pos[0], cube_start_pos[1], 0.90], eef_orientation,
        #     capture_frames=True, iter_folder=temp_folder,
        #     frame_counter=frame_counter, base_pos=robot.base_pos,
        #     state_history=state_history, cube_id=cube_id,
        #     cube_pos_history=cube_pos_history, table_id=table_id,
        #     plane_id=plane_id, tray_id=cylinder_id,
        #     EXCLUDE_TABLE=EXCLUDE_TABLE
        # )
        step1 = move_with_planner(
            planner1, robot, [cube_start_pos[0], cube_start_pos[1], 0.8], eef_orientation,
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )
        step1_history = state_history.copy()

        # Phase 2: Move down to grasp
        print("Phase 2: Moving down to grasp...")
        move_to_pose_dynamic(
            robot, [cube_start_pos[0], cube_start_pos[1], 0.65], eef_orientation,
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )
        planner1.cleanup()
        step2_history = state_history.copy()

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
        step3_history = state_history.copy()

        # Phase 4: Lift cube
        print("Phase 4: Lifting cube...")
        move_to_pose_dynamic(
            robot, [cube_start_pos[0], cube_start_pos[1], 0.8], eef_orientation,
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )
        step4_history = state_history.copy()

        # Phase 5: Move above cylinder
        print("Phase 5: Moving above cylinder target...")
        # Target the top of the cylinder [0.3, 0.3, 0.625 + 0.04] + half cube height [0.025] + safety margin
        cylinder_height = 0.04
        ## old pos 
        # target_drop_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + cylinder_height + CUBE_LENGTH/2 + 0.22]
        ## new pos
        target_drop_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + cylinder_height + CUBE_LENGTH/2 + 0.02]
        print(f"  Target position: [{target_drop_pos[0]:.4f}, {target_drop_pos[1]:.4f}, {target_drop_pos[2]:.4f}]")

        # move_to_pose_dynamic(
        #     robot, target_drop_pos, eef_orientation,
        #     capture_frames=True, iter_folder=temp_folder,
        #     frame_counter=frame_counter, base_pos=robot.base_pos,
        #     state_history=state_history, cube_id=cube_id,
        #     cube_pos_history=cube_pos_history, table_id=table_id,
        #     plane_id=plane_id, tray_id=cylinder_id,
        #     EXCLUDE_TABLE=EXCLUDE_TABLE
        # )
        move_with_planner(
            planner2, robot, target_drop_pos, eef_orientation,
            capture_frames=True, iter_folder=temp_folder,
            frame_counter=frame_counter, base_pos=robot.base_pos,
            state_history=state_history, cube_id=cube_id,
            cube_pos_history=cube_pos_history, table_id=table_id,
            plane_id=plane_id, tray_id=cylinder_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE
        )
        step5_history = state_history.copy()
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
        step6_history = state_history.copy()
        planner2.cleanup()
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
            np.save(os.path.join(final_folder, "step1_history.npy"), step1_history)
            np.save(os.path.join(final_folder, "step2_history.npy"), step2_history)
            np.save(os.path.join(final_folder, "step3_history.npy"), step3_history) 
            np.save(os.path.join(final_folder, "step4_history.npy"), step4_history)
            np.save(os.path.join(final_folder, "step5_history.npy"), step5_history)
            np.save(os.path.join(final_folder, "step6_history.npy"), step6_history)
            
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

def motion_plan_data(
    robot, table_id, plane_id, EXCLUDE_TABLE,
    robot_urdf_path,
    base_save_dir="dataset_mp",
    yaml_config_path="config/planner_config.yaml",
    target_iterations=150,
):
    """
    Motion-planning data collection with update_simulation-based frame capture.

    Per-iteration data saved under  base_save_dir/iter_XXXX/:
        thirdperson_cam00/  rgb / depth / pcd / segmentation
        thirdperson_cam01/  rgb / depth / pcd / segmentation
        thirdperson_cam02/  rgb / depth / pcd / segmentation
        camera_poses/       pose_XXXX.json   (all 3 cam poses per frame)
        agent_pos.npy       robot state  (N, 7) : 6 joints + gripper
        actions.npy         next state   (N, 7)
        state_action.json
        start_configuration.json
        end_configuration.json
    """

    print("\n" + "="*60)
    print("STARTING MOTION PLANNING DATA COLLECTION")
    print(f"Target: {target_iterations} successful trajectories")
    print("="*60 + "\n")

    # Load YAML config once
    try:
        with open(yaml_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print("Warning: planner_config.yaml not found. Using defaults.")
        config_dict = {
            'planner': {'algorithm': 'RRTConnect', 'planning_time': 5.0,
                        'collision_margin': 0.02, 'resolution': 0.005},
            'smoothing': {'enable_smoothing': True, 'smooth_steps': 10,
                          'interpolate_points': 150, 'min_change': 0.01},
            'optimization': {'objective': "PathLength", 'cost_threshold': 1.1},
        }

    p_config  = config_dict['planner']
    obs_config = config_dict.get('obstacles', {})

    robot.reset_posture()

    successful_iterations = 0
    total_attempts        = 0
    failed_attempts       = 0

    while successful_iterations < target_iterations:
        total_attempts += 1

        # ------------------------------------------------------------------
        # 1. Randomise start / goal positions
        # ------------------------------------------------------------------
        const_x_pos = 0.30
        const_y_pos = 0.20
        const_z_pos = 0.625
        # point_A_pos = [const_x_pos,-const_y_pos+0.07, np.random.uniform(const_z_pos+0.25,const_z_pos+0.4)]
        point_A_pos = [const_x_pos,-const_y_pos+0.07,const_z_pos+np.random.uniform(0.2,0.35)]
        # point_B_pos = [const_x_pos, const_y_pos-0.07,np.random.uniform(const_z_pos+0.25,const_z_pos+0.4)]
        point_B_pos = [const_x_pos, const_y_pos-0.07,const_z_pos+np.random.uniform(0.2,0.35)]
        target_orn = p.getQuaternionFromEuler([-1.5708, 0, 1.5708])
        # print(f"p.connectionInfo(): {p.getConnectionInfo()}")
        if p.getConnectionInfo()['connectionMethod'] == 1:  # 1: GUI , 2 : DIRECT3 
            draw_sphere(point_A_pos, radius=0.02, color=[0, 1, 0])
            draw_sphere(point_B_pos, radius=0.02, color=[1, 0, 0])

        # ------------------------------------------------------------------
        # 2. Spawn obstacles
        # ------------------------------------------------------------------
        # obstacle_ids = generate_task_obstacles_old(point_A_pos, point_B_pos, obs_config['num_obstacles'])
        obstacle_ids = generate_simple_obstacle(point_A_pos,point_B_pos)
        # ------------------------------------------------------------------
        # 3. IK solutions for A and B
        # ------------------------------------------------------------------
        planner = RobotOMPLPlanner(
            robot=robot,
            robot_urdf=robot_urdf_path,
            obstacles=[table_id] + obstacle_ids,
            collision_margin=p_config.get('collision_margin', 0.02),
            ignore_base=True,
            config=config_dict,
        )
        planner.set_planner(p_config['algorithm'])

        print(f"\n{'='*60}")
        print(f"ATTEMPT {total_attempts}  (Successful: {successful_iterations}/{target_iterations})")
        print(f"{'='*60}")

        print("Solving IK for B (goal) …")
        goal_config  = solve_ik_collision_free(robot, planner, point_B_pos, target_orn)
        print("Solving IK for A (start) …")
        start_config = solve_ik_collision_free(robot, planner, point_A_pos, target_orn)

        if not start_config or not goal_config:
            print("❌ IK failed for A or B – skipping attempt.")
            for obs_id in obstacle_ids:
                p.removeBody(obs_id)
            planner.cleanup()
            failed_attempts += 1
            robot.reset_posture()
            continue

        # ------------------------------------------------------------------
        # 4. Record start configuration
        # ------------------------------------------------------------------
        start_configuration = {
            "joint_positions":   [float(v) for v in start_config],
            "eef_position":      list(point_A_pos),
            "eef_orientation":   list(target_orn),
            "gripper_state":     float(robot.get_robot_state()[-1]),
        }

        # ------------------------------------------------------------------
        # 5. Plan trajectory A → B
        # ------------------------------------------------------------------
        visualise_eef_traj.prev_pos = None
        print("Planning A → B …")
        success, path = planner.plan(
            start_config, goal_config,
            planning_time=p_config['planning_time'],
        )

        if not success or not path:
            print("❌ Planning failed – no path found.")
            for obs_id in obstacle_ids:
                p.removeBody(obs_id)
            planner.cleanup()
            failed_attempts += 1
            robot.reset_posture()
            continue

        # ------------------------------------------------------------------
        # 6. Temp folder + data-collection state
        # ------------------------------------------------------------------
        temp_folder   = os.path.join(base_save_dir, f"temp_iter_{total_attempts:04d}")
        os.makedirs(temp_folder, exist_ok=True)

        frame_counter = [0]
        state_history = []   # list of (7,) arrays: 6 joints + gripper

        # ------------------------------------------------------------------
        # 7. Execute path with update_simulation capture
        # ------------------------------------------------------------------
        print(f"Executing path ({len(path)} waypoints) …")

        # Common kwargs forwarded to update_simulation / move helpers
        capture_use_3cam = False
        sim_kwargs = dict(
            capture_frames=True,
            iter_folder=temp_folder,
            frame_counter=frame_counter,
            base_pos=robot.base_pos,
            state_history=state_history,
            table_id=table_id,
            plane_id=plane_id,
            EXCLUDE_TABLE=EXCLUDE_TABLE,
            use_3cam=capture_use_3cam,
        )

        collision_occurred = False

        # test_lite6_ompl.visualize_path(robot, path)
        # Restore robot to start_config after visualization preview
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.resetJointState(robot.id, joint_id, start_config[i])

        # planner._snapshot_gripper_pose()
        # planner._apply_frozen_gripper_printing()
        MAX_STEPS_PER_WAYPOINT = 2   # hard cap — tune this down (e.g. 2-3)
        WAYPOINT_THRESHOLD = 0.01
        print(type(path), len(path))
        for i, waypoint in enumerate(path):  # Skip every other waypoint for faster execution (tune as needed)
            for j, joint_id in enumerate(planner.joint_ids):
                p.setJointMotorControl2(
                    planner.robot.id, joint_id, p.POSITION_CONTROL,
                    waypoint[j],
                    force=150.0,
                    maxVelocity=robot.max_velocity,
                    positionGain=0.05,
                    velocityGain=1.0,
                )

            eef_state = p.getLinkState(planner.robot.id, planner.robot.eef_id)
            visualise_eef_traj(eef_state[0])

            for step in range(MAX_STEPS_PER_WAYPOINT):
                # planner._apply_frozen_gripper()

                # --- step simulation + capture one frame ---
                
                update_simulation(1, robot=robot, **sim_kwargs)
                # p.stepSimulation()

                # Collision check
                for obs_id in [table_id] + obstacle_ids:
                    contacts = p.getContactPoints(robot.id, obs_id)
                    if len(contacts) > 0:
                        collision_occurred = True
                        print(f"💥 COLLISION at waypoint {i}!")
                        break
                if collision_occurred:
                    break
                current = [p.getJointState(robot.id, j)[0] for j in planner.joint_ids]
                if np.max(np.abs(np.array(current) - np.array(waypoint))) < WAYPOINT_THRESHOLD:
                    break

            if collision_occurred:
                break

            if i % max(1, len(path) // 10) == 0:
                print(f"  Progress: {i+1}/{len(path)} waypoints, frames: {frame_counter[0]}")
        # FILLING THE GAP IN CASE SOME DISTANCE AFTER THE PLAN IS LEFT
        if not collision_occurred:
            move_to_pose_dynamic(
                robot, point_B_pos, target_orn,
                max_steps=100,
                capture_frames=True, iter_folder=temp_folder,
                frame_counter=frame_counter, base_pos=robot.base_pos,
                state_history=state_history, cube_id=None,
                cube_pos_history=None, table_id=table_id,
                plane_id=plane_id, tray_id=None,
                EXCLUDE_TABLE=EXCLUDE_TABLE,
                use_3cam=capture_use_3cam,
            )
        # ------------------------------------------------------------------
        # 8. Final verification
        # ------------------------------------------------------------------
        time.sleep(0.5)
        final_pos, final_orn = robot.get_current_ee_position()
        pos_err  = np.linalg.norm(np.array(final_pos) - np.array(point_B_pos))
        dot      = np.clip(abs(np.dot(final_orn, target_orn)), 0.0, 1.0)
        orn_err  = math.degrees(2.0 * math.acos(dot))

        reached_goal = (not collision_occurred)
        # 3. Did the robot actually follow the planned path?
        # Compare final joint state vs last waypoint of path
        final_joints = [p.getJointState(robot.id, j)[0] for j in robot.arm_controllable_joints]
        last_waypoint = np.array(path[-1])
        tracking_err = np.max(np.abs(np.array(final_joints) - last_waypoint))
        if tracking_err > 0.15:
            print(f"⚠️  High tracking error: {tracking_err:.4f} rad — saving anyway")

        # ------------------------------------------------------------------
        # 9. Record end configuration
        # ------------------------------------------------------------------
        end_configuration = {
            "joint_positions":   [float(v) for v in [p.getJointState(robot.id, jid)[0]
                                                       for jid in robot.arm_controllable_joints]],
            "eef_position":      list(final_pos),
            "eef_orientation":   list(final_orn),
            "gripper_state":     float(robot.get_robot_state()[-1]),
            "pos_error_m":       float(pos_err),
            "orn_error_deg":     float(orn_err),
            "collision_occurred": collision_occurred,
            "tracking_err_rad": tracking_err,   # ADDED to filter data
        }

        # ------------------------------------------------------------------
        # 10. Cleanup obstacles & planner regardless of outcome
        # ------------------------------------------------------------------
        for obs_id in obstacle_ids:
            p.removeBody(obs_id)

        if not reached_goal:
            reason = "Collision" if collision_occurred else "Missed Target"
            print(f"\n❌ FAILURE: {reason} | pos_err={pos_err*1000:.1f} mm | orn_err={orn_err:.1f}°\n")
            import shutil
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
            failed_attempts += 1
            robot.reset_posture()
            continue

        # ------------------------------------------------------------------
        # 11. SUCCESS – persist data
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! pos_err={pos_err*1000:.1f} mm | frames={frame_counter[0]}")
        print(f"{'='*60}\n")

        final_folder = os.path.join(base_save_dir, f"iter_{successful_iterations:04d}")
        if os.path.exists(final_folder):
            _shutil.rmtree(final_folder)
        os.rename(temp_folder, final_folder)

        # Build state-action arrays
        agent_pos = np.array(state_history)         # (N, 7)

        actions = np.zeros_like(agent_pos)
        if len(agent_pos) > 1:
            actions[:-1] = agent_pos[1:]
            actions[-1]  = agent_pos[-1]

        # Filter near-static frames
        deltas    = actions - agent_pos
        dists     = np.linalg.norm(deltas, axis=1)
        keep_mask = dists >= 0.008
        if len(keep_mask) > 0:
            keep_mask[-1] = True

        original_count = len(agent_pos)
        agent_pos      = agent_pos[keep_mask]
        actions        = actions[keep_mask]

        print(f"  Filtered static frames: {original_count} → {len(agent_pos)}")

        # Prune per-frame files to match filtered states.
        removed_indices = set(np.where(~keep_mask)[0])
        if removed_indices:
            prune_entries = []
            if capture_use_3cam:
                for cam_name in ["thirdperson_cam00", "thirdperson_cam01", "thirdperson_cam02"]:
                    prune_entries += [
                        (os.path.join(final_folder, cam_name, "rgb"),
                         f"{cam_name}_rgb_{{:04d}}.png"),
                        (os.path.join(final_folder, cam_name, "depth"),
                         f"{cam_name}_depth_{{:04d}}.npy"),
                        (os.path.join(final_folder, cam_name, "pcd"),
                         f"{cam_name}_pcd_{{:04d}}.npy"),
                        (os.path.join(final_folder, cam_name, "pcd"),
                         f"{cam_name}_pcd_{{:04d}}.ply"),
                        (os.path.join(final_folder, cam_name, "segmentation"),
                         f"{cam_name}_seg_{{:04d}}.npy"),
                    ]
            else:
                prune_entries += [
                    (os.path.join(final_folder, "third_person", "rgb"), "tp_rgb_{:04d}.png"),
                    (os.path.join(final_folder, "third_person", "depth"), "tp_depth_{:04d}.npy"),
                    (os.path.join(final_folder, "third_person", "pcd"), "tp_pcd_{:04d}.npy"),
                    (os.path.join(final_folder, "third_person", "pcd"), "tp_pcd_{:04d}.ply"),
                    (os.path.join(final_folder, "third_person", "segmentation"), "tp_seg_{:04d}.npy"),
                ]
            prune_entries.append(
                (os.path.join(final_folder, "camera_poses"), "pose_{:04d}.json")
            )

            for dir_path, pattern in prune_entries:
                if not os.path.isdir(dir_path):
                    continue
                ext = pattern.split(".")[-1]
                for idx in removed_indices:
                    fpath = os.path.join(dir_path, pattern.format(idx))
                    if os.path.exists(fpath):
                        os.remove(fpath)
                surviving = sorted(f for f in os.listdir(dir_path) if f.endswith(f".{ext}"))
                for new_idx, old_name in enumerate(surviving):
                    new_name = pattern.format(new_idx)
                    if old_name != new_name:
                        os.rename(
                            os.path.join(dir_path, old_name),
                            os.path.join(dir_path, new_name),
                        )

            print(f"  Pruned per-frame files: kept {len(agent_pos)}/{original_count}")

        # Save start / end configs
        with open(os.path.join(final_folder, "start_configuration.json"), "w") as f:
            json.dump(start_configuration, f, indent=2)
        with open(os.path.join(final_folder, "end_configuration.json"), "w") as f:
            json.dump(end_configuration, f, indent=2)

        # Save camera poses summary (static – same for all frames of this run)
        cam_poses_summary = {
            cam_cfg["name"]: {
                "eye_world":    cam_cfg["eye"],
                "target_world": cam_cfg["target"],
                "up":           cam_cfg["up"],
                "extrinsics":   compute_extrinsics(
                    np.array(cam_cfg["eye"])    - np.array(robot.base_pos),
                    np.array(cam_cfg["target"]) - np.array(robot.base_pos),
                    cam_cfg["up"],
                ),
            }
            for cam_cfg in _MP_CAMERAS
        }
        with open(os.path.join(final_folder, "camera_poses_summary.json"), "w") as f:
            json.dump(cam_poses_summary, f, indent=2)

        # Save numpy arrays
        np.save(os.path.join(final_folder, "agent_pos.npy"), agent_pos)
        np.save(os.path.join(final_folder, "actions.npy"),   actions)

        # Save human-readable text
        np.savetxt(
            os.path.join(final_folder, "agent_pos.txt"), agent_pos, fmt="%.6f",
            header="6 joint angles + normalized gripper (7 dims)",
        )
        np.savetxt(
            os.path.join(final_folder, "actions.txt"), actions, fmt="%.6f",
            header="Absolute target state at t+1 (7 dims)",
        )

        # Save JSON summary
        state_action_data = {
            "agent_pos":          agent_pos.tolist(),
            "action":             actions.tolist(),
            "num_frames":         int(len(agent_pos)),
            "state_dim":          7,
            "success":            True,
            "attempt_number":     total_attempts,
            "success_number":     successful_iterations,
            "point_A":            point_A_pos,
            "point_B":            point_B_pos,
            "final_eef_pos":      list(final_pos),
            "pos_error_m":        float(pos_err),
            "orn_error_deg":      float(orn_err),
            "tracking_err_rad": tracking_err,   # ADDED
            "cameras":            [c["name"] for c in _MP_CAMERAS],
            "state_description":  {"arm_joints": [0,1,2,3,4,5], "gripper": [6]},
            "action_description": "Absolute target state (next state) for each timestep",
        }
        with open(os.path.join(final_folder, "state_action.json"), "w") as f:
            json.dump(state_action_data, f, indent=2)

        print(f"Saved trajectory to: {final_folder}")
        print(f"  Frames: {len(agent_pos)} | States: {agent_pos.shape} | Actions: {actions.shape}")

        successful_iterations += 1

        # ------------------------------------------------------------------
        # 12. Progress report
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"PROGRESS: {successful_iterations}/{target_iterations} "
              f"({100*successful_iterations/target_iterations:.1f}%)  "
              f"| Attempts: {total_attempts} | Failed: {failed_attempts}")
        print(f"{'='*60}\n")

        robot.reset_posture()

    # ----------------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------------
    print("\n" + "="*60)
    print("MOTION PLANNING DATA COLLECTION COMPLETE!")
    print("="*60)
    print(f"Total attempts:               {total_attempts}")
    print(f"Successful trajectories:      {successful_iterations}")
    print(f"Failed / discarded:           {failed_attempts}")
    print(f"Overall success rate:         {100*successful_iterations/total_attempts:.1f}%")
    print("="*60 + "\n")

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

    return None, table_id, plane_id

def main():

    EXCLUDE_TABLE = True

    """
    If excluding table from point clouds, set EXCLUDE_TABLE = True
    Or else False to include table in point clouds
    """

    _, table_id, plane_id = setup_simulation(freq=60, gui=False)
    robot = Lite6Robot([0, 0, 0.62], [0, 0, 0])
    robot_urdf_path = robot.urdf_path
    robot.load()
    # interactive_camera_helper()
    tcp_link_index = -1
    for i in range(p.getNumJoints(robot.id)):
        joint_info = p.getJointInfo(robot.id, i)
        # index 1 is the joint name, index 12 is the child link name
        link_name = joint_info[12].decode('utf-8') 
        if link_name == "tcp":
            tcp_link_index = i
            print(f"Found TCP link '{link_name}' at index {tcp_link_index}")
            break

    if tcp_link_index != -1:
        robot.eef_id = tcp_link_index
    # move_and_grab_cube(robot, table_id, plane_id, EXCLUDE_TABLE=EXCLUDE_TABLE)
    motion_plan_data(
        robot, table_id, plane_id,
        EXCLUDE_TABLE=EXCLUDE_TABLE,
        robot_urdf_path=robot_urdf_path,
        base_save_dir="/scratch3/cross-emb/dataset_mp_poco",
        yaml_config_path="config/planner_config.yaml",
        target_iterations=150,
    )
    p.disconnect()

if __name__ == "__main__":
    main()
