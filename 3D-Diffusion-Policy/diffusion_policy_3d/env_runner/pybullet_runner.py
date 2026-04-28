import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from termcolor import cprint
from diffusion_policy_3d.env import UR5PickPlaceEnv, Lite6MotionPlanningEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import (
    SimpleVideoRecordingWrapper,
)

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util

import open3d as o3d
import numpy as np

import wandb
import cv2
import csv

import os
from datetime import datetime


def save_pointcloud(pc, fname="debug_pc.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.astype(np.float64))
    o3d.io.write_point_cloud(fname, pcd)
    print(f"[PCD] saved to {fname}")


def discretize_gripper_action(gripper_value):
    """
    Discretize continuous gripper action to {-1, 0, 1}
    
    Args:
        gripper_value: Continuous value from policy
    
    Returns:
        Discrete command: -1.0 (open), 0.0 (hold), or 1.0 (close)
    """
    if gripper_value < -0.33:
        return -1.0
    elif gripper_value > 0.33:
        return 1.0
    else:
        return 0.0


class UR5PyBulletRunner(BaseRunner):
    """
    Extended runner with support for 6D (arm only) and 7D (arm + gripper) actions
    
    Key features:
    1. Handles both 6D and 7D action spaces
    2. Gripper actions are discretized to {-1, 0, 1} when present
    3. Skips gripper comparison when action_dim=6
    """

    def __init__(
        self,
        output_dir,
        n_train=10,
        n_test=1,
        max_steps=350,
        n_obs_steps=2,
        n_action_steps=8,
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
        use_gui=False,
        num_points=6000,
        image_size=224,
        use_workspace_crop=True,
        workspace_std=30.0,
        action_dim=7,
        capture_table=False
    ):
        super().__init__(output_dir)

        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.tqdm_interval_sec = tqdm_interval_sec
        self.action_dim = action_dim
        self.num_points = num_points
        
        # NEW: Determine if gripper is included
        self.include_gripper = (action_dim == 7 or action_dim == 13)
        
        cprint(f"Runner initialized with action_dim={action_dim}", "cyan")
        cprint(f"Include gripper: {self.include_gripper}", "cyan")

        # Environment factory function
        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    UR5PickPlaceEnv(
                        use_gui=use_gui,
                        num_points=num_points,
                        image_size=image_size,
                        use_workspace_crop=use_workspace_crop,
                        workspace_std=workspace_std,
                        action_dim=self.action_dim,
                        capture_table=capture_table,
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method="sum",
            )

        self.env_test = env_fn()

    def run(self, policy: BasePolicy, dataset=None):
        device = policy.device
        all_returns_test = []
        all_success_rates_test = []
        all_collision_rates_test = []
        saved_local_videos = []
        final_goal_dists = []

        cam0_first_episode = []
        cam1_first_episode = []

        for episode_id in range(self.episode_test):
            # Try to get cube position from dataset if available
            cube_start_pos = None
            cube_start_orn = None
            if dataset is not None and hasattr(dataset, 'get_episode_cube_start_pos') and episode_id < dataset.replay_buffer.n_episodes:
                try:
                    cube_info = dataset.get_episode_cube_start_pos(episode_id)
                    if cube_info is not None:
                        cube_start_pos = cube_info[:3]
                        cube_start_orn = cube_info[3:7]
                except:
                    pass

            if cube_start_pos is not None:
                obs = self.env_test.reset(cube_start_pos=cube_start_pos, cube_start_orn=cube_start_orn)
            else:
                obs = self.env_test.reset()
            
            policy.reset()

            reward_sum = 0.0
            done = False
            final_goal_dist = 0.0

            collect_frames = self.save_local_videos or (self.log_wandb_videos and episode_id == 0)
            episode_cam0_frames = []
            episode_cam1_frames = []
            if collect_frames:
                episode_cam0_frames.append(self.env_test.env.render_camera(0))
                episode_cam1_frames.append(self.env_test.env.render_camera(1))

            while not done:
                policy_obs = self._to_policy_obs(obs, policy, device)

                with torch.no_grad():
                    action_dict = policy.predict_action(policy_obs)
                action = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())["action"]

                if action.ndim == 3:
                    action = action.squeeze(0)
                
                action_seq = action[: self.n_action_steps]
                obs, reward, done, info = self.env_test.step(action_seq)
                reward_sum += float(reward)
                
                # Check goal dist (cube to tray center)
                if hasattr(self.env_test.env, 'tray_pos'):
                    tray_pos = self.env_test.env.tray_pos
                    import pybullet as p
                    if self.env_test.env.cube_id is not None:
                        cube_pos, _ = p.getBasePositionAndOrientation(self.env_test.env.cube_id)
                        final_goal_dist = np.linalg.norm(np.array(cube_pos[:2]) - np.array(tray_pos[:2]))

                if collect_frames:
                    episode_cam0_frames.append(self.env_test.env.render_camera(0))
                    episode_cam1_frames.append(self.env_test.env.render_camera(1))

            all_returns_test.append(reward_sum)
            all_success_rates_test.append(1.0 if self.env_test.env.is_success() else 0.0)
            all_collision_rates_test.append(1.0 if (hasattr(self.env_test.env, 'collision_flag') and self.env_test.env.collision_flag) else 0.0)
            final_goal_dists.append(final_goal_dist)

            if self.log_wandb_videos and episode_id == 0 and len(episode_cam0_frames) > 0:
                cam0_first_episode = self._annotate_frames_with_final_error(episode_cam0_frames, final_goal_dist)
                cam1_first_episode = self._annotate_frames_with_final_error(episode_cam1_frames, final_goal_dist)

            if self.save_local_videos and len(episode_cam0_frames) > 0:
                if self.save_all_local_episodes or (episode_id == 0):
                    saved_paths = self._save_local_episode_videos(
                        episode_id=episode_id,
                        cam0_frames=episode_cam0_frames,
                        cam1_frames=episode_cam1_frames,
                        collided=(hasattr(self.env_test.env, 'collision_flag') and self.env_test.env.collision_flag),
                        final_goal_dist=final_goal_dist,
                    )
                    saved_local_videos.extend(saved_paths)

            print(f"Lite6 PickPlace Episode {episode_id}: Reward={reward_sum:.2f}, Success={self.env_test.env.is_success()}")

        sr_mean = float(np.mean(all_success_rates_test)) if len(all_success_rates_test) > 0 else 0.0
        returns_mean = float(np.mean(all_returns_test)) if len(all_returns_test) > 0 else 0.0
        
        log_data = {
            "mean_success_rates_test": sr_mean,
            "mean_returns_test": returns_mean,
            "test_mean_score": sr_mean,
        }

        if self.log_wandb_videos and len(cam0_first_episode) > 0:
            import wandb
            cam0 = np.stack(cam0_first_episode, axis=0)
            cam1 = np.stack(cam1_first_episode, axis=0)
            merged = np.concatenate([cam0, cam1], axis=2).transpose(0, 3, 1, 2)
            log_data["sim_video_cam01_merged_test"] = wandb.Video(
                merged.astype(np.uint8), fps=self.fps, format="mp4"
            )

        return log_data
        self.episode_test = n_test
        self.logger_util_test = logger_util.LargestKRecorder(K=3)

    def run(self, policy: BasePolicy, dataset=None):
        """
        Run evaluation using validation dataset cube positions
        
        Args:
            policy: Policy to evaluate
            dataset: Validation dataset to get cube positions from
        """
        device = policy.device

        # Check what keys the policy normalizer expects
        normalizer_keys = list(policy.normalizer.params_dict.keys())
        cprint(f"Policy expects observation keys: {normalizer_keys}", "yellow")

        all_returns_test = []
        all_success_rates_test = []

        log_dir = os.path.join(self.output_dir, "action_comparisons")
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.include_gripper:
            log_file_path = os.path.join(log_dir, f"gripper_gt_vs_pred_{timestamp}.txt")
        else:
            log_file_path = os.path.join(log_dir, f"arm_only_gt_vs_pred_{timestamp}.txt")
        
        cprint(f"Logging to: {log_file_path}", "cyan")

        cprint("=" * 50, "cyan")
        cprint(
            "Running on TEST environment with VALIDATION dataset cube positions", "cyan"
        )
        cprint("=" * 50, "cyan")

        if dataset is not None:
            val_episode_indices = np.where(~dataset.train_mask)[0]
            n_val_episodes = len(val_episode_indices)
            cprint(f"Validation dataset has {n_val_episodes} episodes", "cyan")
            cprint(f"Running {self.episode_test} test episodes", "cyan")
            if n_val_episodes < self.episode_test:
                cprint(
                    f"WARNING: Only {n_val_episodes} validation episodes available, but running {self.episode_test} test episodes",
                    "yellow",
                )
                cprint(
                    f"Episodes {n_val_episodes}-{self.episode_test-1} will use random cube positions",
                    "yellow",
                )

        for episode_id in range(self.episode_test):
            # Get cube start position from validation dataset if available
            cube_start_pos = None
            cube_start_orn = None
            gt_actions = None

            if dataset is not None:
                try:
                    # Get the episode index from validation set
                    val_episode_indices = np.where(~dataset.train_mask)[0]

                    if episode_id < len(val_episode_indices):
                        val_ep_idx = val_episode_indices[episode_id]
                        cube_pos_full = dataset.get_episode_cube_start_pos(val_ep_idx)

                        # Extract position (first 3 elements) and orientation (last 4 elements)
                        cube_start_pos = cube_pos_full[:3].tolist()
                        cube_start_orn = cube_pos_full[3:7].tolist()
                        cprint(
                            f"Episode {episode_id}: Using cube position from val dataset: {cube_start_pos}",
                            "green",
                        )

                        # Get ground truth actions for this episode
                        episode_data = dataset.get_episode(val_ep_idx)
                        gt_actions = episode_data['action']  # Shape: (T, action_dim)
                        cprint(
                            f"Shape of episode action data: {gt_actions.shape}",
                            "green",
                        )

                    else:
                        cprint(
                            f"Episode {episode_id}: No more validation episodes, using random position",
                            "yellow",
                        )
                        cube_start_pos = None
                        cube_start_orn = None
                        gt_actions = None

                except Exception as e:
                    cprint(
                        f"Episode {episode_id}: Could not load cube position from dataset: {e}",
                        "red",
                    )
                    cprint("Falling back to random cube position", "yellow")
                    cube_start_pos = None
                    cube_start_orn = None

            # Open log file in append mode
            with open(log_file_path, 'a') as log_file:
                # Write episode header
                log_file.write("="*100 + "\n")
                log_file.write(f"EPISODE {episode_id}\n")
                log_file.write("="*100 + "\n")
                if cube_start_pos:
                    log_file.write(f"Cube start position: {cube_start_pos}\n")
                    log_file.write(f"Cube start orientation: {cube_start_orn}\n")
                if gt_actions is not None:
                    log_file.write(f"GT trajectory length: {len(gt_actions)} timesteps\n")
                log_file.write(f"n_action_steps: {self.n_action_steps}\n")
                log_file.write(f"n_obs_steps: {self.n_obs_steps}\n")
                log_file.write(f"action_dim: {self.action_dim}\n")
                log_file.write(f"include_gripper: {self.include_gripper}\n")
                log_file.write("\n")

            # Reset environment with specified or random cube position
            if cube_start_pos is not None:
                self.env_test.env.env.cube_start_pos = cube_start_pos
                self.env_test.env.env.cube_start_orn = cube_start_orn
                obs = self.env_test.reset(cube_start_pos=cube_start_pos, cube_start_orn=cube_start_orn)
            else:
                cprint(
                    f"Episode {episode_id}: Using random cube position TO RESET",
                    "yellow",
                )
                obs = self.env_test.reset()

            # Debug: print shapes on first episode
            if episode_id == 0:
                cprint(f"Environment provides keys: {list(obs.keys())}", "yellow")
                for key, val in obs.items():
                    if isinstance(val, np.ndarray):
                        cprint(
                            f"  {key}: shape={val.shape}, dtype={val.dtype}", "yellow"
                        )
                cprint(f"n_obs_steps: {self.n_obs_steps}", "yellow")

            policy.reset()

            reward_sum = 0.0
            done = False

            # Storage for this episode's comparisons
            episode_gt_actions = []
            episode_pred_actions = []
            episode_pred_actions_discrete = []

            gt_timestep = 0  

            for step_id in range(self.max_steps):
                # Deep copy the observation to avoid reference issues
                np_obs_dict = {}
                for key, val in obs.items():
                    if isinstance(val, np.ndarray):
                        np_obs_dict[key] = val.copy()
                    else:
                        np_obs_dict[key] = val

                # Debug point cloud shape before processing
                if step_id == 0 and episode_id == 0:
                    pc_raw = np_obs_dict["point_cloud"]
                    cprint(f"Raw point cloud shape: {pc_raw.shape}", "cyan")

                    # Handle different observation shapes
                    if pc_raw.ndim == 3:
                        # Stacked observations: (n_obs_steps, num_points, 3)
                        cprint(f"Detected stacked observations: {pc_raw.shape}", "cyan")
                        pc_for_debug = pc_raw[-1]
                    elif pc_raw.ndim == 2:
                        # Single observation: (num_points, 3)
                        pc_for_debug = pc_raw
                    else:
                        cprint(
                            f"WARNING: Unexpected point cloud shape: {pc_raw.shape}",
                            "red",
                        )
                        pc_for_debug = pc_raw.reshape(-1, 3)

                    save_pointcloud(pc_for_debug, "env_pc.ply")
                    cprint(
                        f"Saved debug point cloud with shape: {pc_for_debug.shape}",
                        "cyan",
                    )

                # Convert to torch tensors
                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device, non_blocking=True),
                )

                # Build policy observation dict with proper handling of observation history
                policy_obs = {}
                for key in policy.normalizer.params_dict.keys():
                    if key == "action":
                        continue

                    if key in obs_dict:
                        tensor = obs_dict[key]

                        # Handle stacked vs single observations
                        if key == "point_cloud":
                            if tensor.ndim == 2:
                                cprint(
                                    f"WARNING: point_cloud not stacked, repeating {self.n_obs_steps} times",
                                    "yellow",
                                )
                                tensor = tensor.unsqueeze(0).repeat(
                                    self.n_obs_steps, 1, 1
                                )
                            policy_obs[key] = tensor.unsqueeze(0)

                        elif key == "agent_pos":
                            if tensor.ndim == 1:
                                cprint(
                                    f"WARNING: agent_pos not stacked, repeating {self.n_obs_steps} times",
                                    "yellow",
                                )
                                tensor = tensor.unsqueeze(0).repeat(self.n_obs_steps, 1)
                            policy_obs[key] = tensor.unsqueeze(0)

                        elif key == "image":
                            if tensor.ndim == 3:
                                cprint(
                                    f"WARNING: image not stacked, repeating {self.n_obs_steps} times",
                                    "yellow",
                                )
                                tensor = tensor.unsqueeze(0).repeat(
                                    self.n_obs_steps, 1, 1, 1
                                )
                            policy_obs[key] = tensor.unsqueeze(0)
                        else:
                            policy_obs[key] = tensor.unsqueeze(0)
                    else:
                        cprint(
                            f"WARNING: Policy expects key '{key}' but not in observation!",
                            "red",
                        )

                # Debug shapes before policy prediction
                if step_id == 0 and episode_id == 0:
                    cprint("Policy input shapes:", "cyan")
                    for key, val in policy_obs.items():
                        cprint(f"  {key}: {val.shape}", "cyan")

                with torch.no_grad():
                    action_dict = policy.predict_action(policy_obs)

                # Extract action - handle both single and multi-step actions
                action = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())[
                    "action"
                ]

                # Debug action shape
                if step_id == 0 and episode_id == 0:
                    cprint(f"Raw action shape: {action.shape}", "cyan")

                # MultiStepWrapper expects (n_action_steps, action_dim)
                if action.ndim == 3:
                    action = action.squeeze(0)
                elif action.ndim == 2 and action.shape[0] == 1:
                    action = action.squeeze(0)

                # Final check
                if step_id == 0 and episode_id == 0:
                    cprint(f"Final action shape for env.step(): {action.shape}", "cyan")

                # CRITICAL: Discretize gripper actions if included
                action_discrete = action.copy()
                if self.include_gripper:
                    for i in range(len(action_discrete)):
                        gripper_idx = 6 if action_discrete.shape[1] == 7 else 12
                        gripper_continuous = action_discrete[i, gripper_idx]
                        gripper_discrete = discretize_gripper_action(gripper_continuous)
                        action_discrete[i, gripper_idx] = gripper_discrete

                if gt_actions is not None:
                    # Get the corresponding GT actions that will be executed
                    gt_start_idx = gt_timestep
                    gt_end_idx = min(gt_timestep + self.n_action_steps, len(gt_actions))
                    
                    if gt_start_idx < len(gt_actions):
                        gt_actions_to_execute = gt_actions[gt_start_idx:gt_end_idx]
                        pred_actions_continuous = action[:len(gt_actions_to_execute)]
                        pred_actions_discrete = action_discrete[:len(gt_actions_to_execute)]
                        
                        # Log comparison based on whether gripper is included
                        if self.include_gripper:
                            # Extract gripper values (index 6 or 12)
                            gripper_idx = 6 if gt_actions.shape[1] == 7 else 12
                            gt_grippers = gt_actions_to_execute[:, gripper_idx]
                            pred_grippers_continuous = pred_actions_continuous[:, gripper_idx]
                            pred_grippers_discrete = pred_actions_discrete[:, gripper_idx]
                            
                            for i in range(len(gt_grippers)):
                                actual_gt_timestep = gt_start_idx + i
                        
                                with open(log_file_path, 'a') as log_file:
                                    log_file.write(f"  Action [{i}] - GT Timestep {actual_gt_timestep}:\n")
                                    log_file.write(f"    GT Gripper:        {gt_grippers[i]:.6f}\n")
                                    log_file.write(f"    Pred Continuous:   {pred_grippers_continuous[i]:.6f}\n")
                                    log_file.write(f"    Pred Discrete:     {pred_grippers_discrete[i]:.6f}\n")
                                    log_file.write(f"    Diff (GT-Disc):    {abs(gt_grippers[i] - pred_grippers_discrete[i]):.6f}\n")
                                    log_file.write("\n")
                        else:
                            # Log arm joints only (no gripper)
                            for i in range(len(gt_actions_to_execute)):
                                actual_gt_timestep = gt_start_idx + i
                                gt_arm = gt_actions_to_execute[i, :6]
                                pred_arm_continuous = pred_actions_continuous[i, :6]
                                arm_mae = np.mean(np.abs(gt_arm - pred_arm_continuous))
                                
                                with open(log_file_path, 'a') as log_file:
                                    log_file.write(f"  Action [{i}] - GT Timestep {actual_gt_timestep}:\n")
                                    log_file.write(f"    GT Arm Joints:     {gt_arm}\n")
                                    log_file.write(f"    Pred Arm Joints:   {pred_arm_continuous}\n")
                                    log_file.write(f"    MAE (Arm):         {arm_mae:.6f}\n")
                                    log_file.write("\n")
                        
                        # Store for episode summary
                        episode_gt_actions.extend(gt_actions_to_execute.tolist())
                        episode_pred_actions.extend(pred_actions_continuous.tolist())
                        episode_pred_actions_discrete.extend(pred_actions_discrete.tolist())
                    
                    # Update GT timestep counter
                    gt_timestep += self.n_action_steps

                # Execute actions (discretized if gripper is included)
                obs, reward, done, info = self.env_test.step(action_discrete)

                reward_sum += reward
                done = np.all(done)

                if done:
                    break

            # Episode summary
            if len(episode_gt_actions) > 0 and len(episode_pred_actions_discrete) > 0:
                gt_arr = np.array(episode_gt_actions)
                pred_continuous_arr = np.array(episode_pred_actions)
                pred_discrete_arr = np.array(episode_pred_actions_discrete)
                
                if self.include_gripper:
                    # Gripper-specific metrics
                    gripper_idx = 6 if gt_arr.shape[1] == 7 else 12
                    gt_grippers = gt_arr[:, gripper_idx]
                    pred_continuous_grippers = pred_continuous_arr[:, gripper_idx]
                    pred_discrete_grippers = pred_discrete_arr[:, gripper_idx]
                    
                    mae_gt_continuous = np.mean(np.abs(gt_grippers - pred_continuous_grippers))
                    mae_gt_discrete = np.mean(np.abs(gt_grippers - pred_discrete_grippers))
                    
                    # Console output
                    cprint(f"\n{'='*80}", "magenta")
                    cprint(f"Episode {episode_id} Gripper Summary:", "magenta")
                    cprint(f"  Total actions compared: {len(gt_arr)}", "magenta")
                    cprint(f"  MAE (GT vs Pred Continuous): {mae_gt_continuous:.6f}", "magenta")
                    cprint(f"  MAE (GT vs Pred Discrete):   {mae_gt_discrete:.6f}", "magenta")
                    cprint(f"  GT Range:        [{gt_grippers.min():.6f}, {gt_grippers.max():.6f}]", "magenta")
                    cprint(f"  Pred Cont Range: [{pred_continuous_grippers.min():.6f}, {pred_continuous_grippers.max():.6f}]", "magenta")
                    cprint(f"  Pred Disc Range: [{pred_discrete_grippers.min():.6f}, {pred_discrete_grippers.max():.6f}]", "magenta")
                    cprint(f"{'='*80}\n", "magenta")
                    
                    # File output
                    with open(log_file_path, 'a') as log_file:
                        log_file.write("\n" + "="*80 + "\n")
                        log_file.write(f"EPISODE {episode_id} SUMMARY\n")
                        log_file.write("="*80 + "\n")
                        log_file.write(f"Total actions compared: {len(gt_arr)}\n")
                        log_file.write(f"MAE (GT vs Pred Continuous): {mae_gt_continuous:.6f}\n")
                        log_file.write(f"MAE (GT vs Pred Discrete):   {mae_gt_discrete:.6f}\n")
                        log_file.write(f"GT Range:        [{gt_grippers.min():.6f}, {gt_grippers.max():.6f}]\n")
                        log_file.write(f"Pred Cont Range: [{pred_continuous_grippers.min():.6f}, {pred_continuous_grippers.max():.6f}]\n")
                        log_file.write(f"Pred Disc Range: [{pred_discrete_grippers.min():.6f}, {pred_discrete_grippers.max():.6f}]\n")
                        log_file.write(f"Reward:          {reward_sum:.2f}\n")
                        log_file.write(f"Success:         {self.env_test.env.is_success()}\n")
                        log_file.write("="*80 + "\n\n\n")
                else:
                    # Arm-only metrics (no gripper)
                    gt_arm = gt_arr[:, :6]
                    pred_arm = pred_continuous_arr[:, :6]
                    
                    mae_per_joint = np.mean(np.abs(gt_arm - pred_arm), axis=0)
                    mae_overall = np.mean(np.abs(gt_arm - pred_arm))
                    
                    # Console output
                    cprint(f"\n{'='*80}", "magenta")
                    cprint(f"Episode {episode_id} Arm Joints Summary:", "magenta")
                    cprint(f"  Total actions compared: {len(gt_arr)}", "magenta")
                    cprint(f"  MAE (Overall):          {mae_overall:.6f}", "magenta")
                    cprint(f"  MAE per joint:          {mae_per_joint}", "magenta")
                    cprint(f"{'='*80}\n", "magenta")
                    
                    # File output
                    with open(log_file_path, 'a') as log_file:
                        log_file.write("\n" + "="*80 + "\n")
                        log_file.write(f"EPISODE {episode_id} SUMMARY\n")
                        log_file.write("="*80 + "\n")
                        log_file.write(f"Total actions compared: {len(gt_arr)}\n")
                        log_file.write(f"MAE (Overall):          {mae_overall:.6f}\n")
                        log_file.write(f"MAE per joint:          {mae_per_joint}\n")
                        log_file.write(f"Reward:                 {reward_sum:.2f}\n")
                        log_file.write(f"Success:                {self.env_test.env.is_success()}\n")
                        log_file.write("="*80 + "\n\n\n")

            all_returns_test.append(reward_sum)
            all_success_rates_test.append(self.env_test.env.is_success())

            cprint(
                f"Test Episode {episode_id}: "
                f"Reward={reward_sum:.2f}, "
                f"Success={self.env_test.env.is_success()}",
                "yellow",
            )

        # ---- Metrics ----
        SR_mean_test = np.mean(all_success_rates_test)
        returns_mean_test = np.mean(all_returns_test)

        self.logger_util_test.record(SR_mean_test)

        log_data = {
            "mean_success_rates_test": SR_mean_test,
            "mean_returns_test": returns_mean_test,
            "SR_test_L3": self.logger_util_test.average_of_largest_K(),
            "test_mean_score": SR_mean_test,
        }

        with open(log_file_path, 'a') as log_file:
            log_file.write("\n\n" + "="*100 + "\n")
            log_file.write("FINAL SUMMARY - ALL EPISODES\n")
            log_file.write("="*100 + "\n")
            log_file.write(f"Mean Success Rate: {SR_mean_test:.3f}\n")
            log_file.write(f"Mean Return:       {returns_mean_test:.3f}\n")
            log_file.write(f"Total Episodes:    {self.episode_test}\n")
            log_file.write("="*100 + "\n")

        cprint("=" * 50, "green")
        cprint(f"Test - Mean SR: {SR_mean_test:.3f}, Mean Return: {returns_mean_test:.3f}", "green")
        cprint(f"Log saved to: {log_file_path}", "green")
        cprint("=" * 50, "green")

        # ---- Video logging ----
        try:
            videos_test = self.env_test.env.get_video()
            if len(videos_test.shape) == 5:
                videos_test = videos_test[:, 0]

            log_data["sim_video_test"] = wandb.Video(
                videos_test, fps=self.fps, format="mp4"
            )
            cprint("✓ Test video captured", "cyan")

        except Exception as e:
            cprint(f"⚠ Video capture failed: {e}", "yellow")

        try:
            _ = self.env_test.reset()
        except:
            pass

        return log_data


class Lite6BaseRunner(BaseRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    @staticmethod
    def _last_info_value(value):
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        if arr.size == 0:
            return None
        return arr.reshape(-1)[-1].item()

    def _to_policy_obs(self, stacked_obs, policy, device):
        policy_obs = {}
        for key in policy.normalizer.params_dict.keys():
            if key == "action":
                continue
            if key not in stacked_obs:
                continue
            tensor = torch.from_numpy(stacked_obs[key]).to(device=device, non_blocking=True)
            # Some policies expect (1, T, D)
            policy_obs[key] = tensor.unsqueeze(0)
        return policy_obs

    def _write_rgb_video(self, frames, output_path):
        if len(frames) == 0:
            return False
        h, w, _ = frames[0].shape
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1, int(self.fps)),
            (w, h),
        )
        if not writer.isOpened():
            return False
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return True

    def _overlay_text(self, frame, text, org=(12, 28)):
        annotated = frame.copy()
        cv2.putText(
            annotated,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated

    def _annotate_frames_with_final_error(self, frames, final_goal_dist):
        if len(frames) == 0 or final_goal_dist is None:
            return frames
        text = f"Final error: {final_goal_dist:.3f} m"
        return [self._overlay_text(frame, text) for frame in frames]

    def _save_local_episode_videos(self, episode_id, cam0_frames, cam1_frames, collided, final_goal_dist=None):
        out_dir = os.path.join(self.output_dir, self.local_video_dir)
        os.makedirs(out_dir, exist_ok=True)
        collision_tag = "collision" if collided else "non_collision"

        cam0_frames = self._annotate_frames_with_final_error(cam0_frames, final_goal_dist)
        cam1_frames = self._annotate_frames_with_final_error(cam1_frames, final_goal_dist)

        if self.merge_cams_side_by_side:
            merged_frames = [
                np.concatenate([f0, f1], axis=1)
                for f0, f1 in zip(cam0_frames, cam1_frames)
            ]
            merged_path = os.path.join(
                out_dir,
                f"episode_{episode_id:03d}_{collision_tag}_cam01_merged.mp4",
            )
            self._write_rgb_video(merged_frames, merged_path)
            return [merged_path]

        cam0_path = os.path.join(out_dir, f"episode_{episode_id:03d}_{collision_tag}_cam0.mp4")
        cam1_path = os.path.join(out_dir, f"episode_{episode_id:03d}_{collision_tag}_cam1.mp4")
        self._write_rgb_video(cam0_frames, cam0_path)
        self._write_rgb_video(cam1_frames, cam1_path)
        return [cam0_path, cam1_path]


class Lite6MotionPlanningRunner(Lite6BaseRunner):
    def __init__(
        self,
        output_dir,
        n_test=5,
        max_steps=350,
        n_obs_steps=1,
        n_action_steps=12,
        fps=10,
        use_gui=False,
        num_points=5000,
        image_size=224,
        action_dim=7,
        collision_terminate=False,
        collision_distance_threshold=0.01,
        success_threshold=0.03,
        max_steps_per_waypoint=10,
        waypoint_threshold=0.01,
        sim_freq=60,
        obstacle_max_top_z=None,
        log_wandb_videos=True,
        save_local_videos=False,
        save_all_local_episodes=False,
        merge_cams_side_by_side=True,
        local_video_dir="rollout_videos",
    ):
        super().__init__(output_dir)
        self.episode_test = n_test
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.action_dim = action_dim
        self.max_steps_per_waypoint = max_steps_per_waypoint
        self.sim_freq = sim_freq
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.log_wandb_videos = log_wandb_videos
        self.save_local_videos = save_local_videos
        self.save_all_local_episodes = save_all_local_episodes
        self.merge_cams_side_by_side = merge_cams_side_by_side
        self.local_video_dir = local_video_dir

        self.env_test = MultiStepWrapper(
            SimpleVideoRecordingWrapper(
                Lite6MotionPlanningEnv(
                    use_gui=use_gui,
                    num_points=num_points,
                    image_size=image_size,
                    action_dim=action_dim,
                    max_steps=max_steps,
                    collision_terminate=collision_terminate,
                    collision_distance_threshold=collision_distance_threshold,
                    success_threshold=success_threshold,
                    obs_config={"max_top_z": obstacle_max_top_z} if obstacle_max_top_z is not None else None,
                    max_steps_per_waypoint=max_steps_per_waypoint,
                    waypoint_threshold=waypoint_threshold,
                    sim_freq=sim_freq,
                )
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
            reward_agg_method="sum",
        )


    def run(self, policy: BasePolicy, dataset=None):
        device = policy.device
        all_returns_test = []
        all_success_rates_test = []
        all_collision_rates_test = []
        saved_local_videos = []
        final_goal_dists = []
        obstacle_top_zs = []

        cam0_first_episode = []
        cam1_first_episode = []
        base_env = self.env_test.env.env
        diag_dir = os.path.join(self.output_dir, "rollout_diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        diag_path = os.path.join(diag_dir, f"motion_rollout_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        diag_fields = [
            "episode_id", "dataset_episode_idx", "policy_chunk_idx", "chunk_action_idx",
            "rollout_action_idx", "env_step", "video_time_sec", "nominal_sim_time_sec",
            "collision", "collision_ever", "done", "success", "goal_dist",
            "obstacle_top_z",
        ]
        diag_fields += [f"action_{i}" for i in range(self.action_dim)]
        diag_fields += [f"state_{i}" for i in range(self.action_dim)]
        diag_fields += [f"gt_action_{i}" for i in range(self.action_dim)]
        val_episode_indices = None
        if dataset is not None and hasattr(dataset, "train_mask"):
            val_episode_indices = np.where(~dataset.train_mask)[0]

        with open(diag_path, "w", newline="") as diag_file:
            diag_writer = csv.DictWriter(diag_file, fieldnames=diag_fields)
            diag_writer.writeheader()

            for episode_id in range(self.episode_test):
                reset_kwargs = {}
                dataset_episode_idx = None
                gt_episode_actions = None
                if dataset is not None and hasattr(dataset, "get_episode_eval_setup"):
                    try:
                        if val_episode_indices is not None and episode_id < len(val_episode_indices):
                            dataset_episode_idx = int(val_episode_indices[episode_id])
                        else:
                            dataset_episode_idx = episode_id
                        if dataset_episode_idx < dataset.replay_buffer.n_episodes:
                            setup = dataset.get_episode_eval_setup(dataset_episode_idx)
                            reset_kwargs = {
                                "start_joint": setup["start_joint"],
                                "start_eef_xyz": setup["start_eef_xyz"],
                                "goal_eef_xyz": setup["goal_eef_xyz"],
                                "start_gripper": setup["start_gripper"],
                            }
                            ep_start = 0 if dataset_episode_idx == 0 else int(dataset.episode_ends[dataset_episode_idx - 1])
                            ep_end = int(dataset.episode_ends[dataset_episode_idx])
                            gt_episode_actions = dataset.replay_buffer["action"][ep_start:ep_end, :self.action_dim]
                    except Exception as exc:
                        cprint(f"Motion-plan eval falling back to random reset: {exc}", "yellow")
                        reset_kwargs = {}
                        dataset_episode_idx = None
                        gt_episode_actions = None

                obs = self.env_test.reset(**reset_kwargs)
                policy.reset()

                reward_sum = 0.0
                done = False
                final_goal_dist = None
                policy_chunk_idx = 0
                rollout_action_idx = 0

                collect_frames = self.save_local_videos or (self.log_wandb_videos and episode_id == 0)
                episode_cam0_frames = []
                episode_cam1_frames = []
                if collect_frames:
                    episode_cam0_frames.append(base_env.render_camera(0))
                    episode_cam1_frames.append(base_env.render_camera(1))

                while not done:
                    policy_obs = self._to_policy_obs(obs, policy, device)

                    with torch.no_grad():
                        action_dict = policy.predict_action(policy_obs)
                    action = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())["action"]

                    if action.ndim == 3:
                        action = action.squeeze(0)
                    elif action.ndim == 1:
                        action = action[None, :]

                    action_seq = action[: self.n_action_steps]
                    for chunk_action_idx, act in enumerate(action_seq):
                        if done:
                            break
                        obs, reward, done, info = self.env_test.step(act[None, :])
                        reward_sum += float(reward)

                        info_goal_dist = self._last_info_value(info.get("goal_dist", final_goal_dist))
                        if info_goal_dist is not None:
                            final_goal_dist = float(info_goal_dist)

                        obstacle_top_z = getattr(base_env, "last_obstacle_info", {}).get("top_z", None)
                        state = np.asarray(obs["agent_pos"][-1], dtype=np.float32)
                        row = {
                            "episode_id": episode_id,
                            "dataset_episode_idx": "" if dataset_episode_idx is None else dataset_episode_idx,
                            "policy_chunk_idx": policy_chunk_idx,
                            "chunk_action_idx": chunk_action_idx,
                            "rollout_action_idx": rollout_action_idx,
                            "env_step": int(base_env.current_step),
                            "video_time_sec": float(base_env.current_step) / float(max(1, self.fps)),
                            "nominal_sim_time_sec": (
                                float(base_env.current_step * self.max_steps_per_waypoint)
                                / float(max(1, self.sim_freq))
                            ),
                            "collision": bool(self._last_info_value(info.get("collision", False))),
                            "collision_ever": bool(base_env.collision_flag),
                            "done": bool(done),
                            "success": bool(base_env.is_success()),
                            "goal_dist": "" if final_goal_dist is None else final_goal_dist,
                            "obstacle_top_z": "" if obstacle_top_z is None else float(obstacle_top_z),
                        }
                        for i in range(self.action_dim):
                            row[f"action_{i}"] = float(act[i])
                            row[f"state_{i}"] = float(state[i])
                            if gt_episode_actions is not None and rollout_action_idx < len(gt_episode_actions):
                                row[f"gt_action_{i}"] = float(gt_episode_actions[rollout_action_idx, i])
                            else:
                                row[f"gt_action_{i}"] = ""
                        diag_writer.writerow(row)
                        diag_file.flush()

                        rollout_action_idx += 1
                        if collect_frames:
                            episode_cam0_frames.append(base_env.render_camera(0))
                            episode_cam1_frames.append(base_env.render_camera(1))

                    policy_chunk_idx += 1

                all_returns_test.append(reward_sum)
                all_success_rates_test.append(1.0 if base_env.is_success() else 0.0)
                all_collision_rates_test.append(1.0 if base_env.collision_flag else 0.0)
                final_goal_dists.append(final_goal_dist)
                obstacle_top_z = getattr(base_env, "last_obstacle_info", {}).get("top_z", None)
                if obstacle_top_z is not None:
                    obstacle_top_zs.append(float(obstacle_top_z))

                if self.log_wandb_videos and episode_id == 0 and len(episode_cam0_frames) > 0:
                    cam0_first_episode = self._annotate_frames_with_final_error(episode_cam0_frames, final_goal_dist)
                    cam1_first_episode = self._annotate_frames_with_final_error(episode_cam1_frames, final_goal_dist)

                if self.save_local_videos and len(episode_cam0_frames) > 0:
                    if self.save_all_local_episodes or (episode_id == 0):
                        saved_paths = self._save_local_episode_videos(
                            episode_id=episode_id,
                            cam0_frames=episode_cam0_frames,
                            cam1_frames=episode_cam1_frames,
                            collided=base_env.collision_flag,
                            final_goal_dist=final_goal_dist,
                        )
                        saved_local_videos.extend(saved_paths)

                cprint(
                    f"Lite6 Test Episode {episode_id}: Reward={reward_sum:.2f}, Success={base_env.is_success()}, Collision={base_env.collision_flag}, GoalDist={final_goal_dist}, ObstacleTopZ={obstacle_top_z}",
                    "yellow",
                )

        cprint(f"Saved motion rollout diagnostics to: {diag_path}", "cyan")

        sr_mean = float(np.mean(all_success_rates_test)) if len(all_success_rates_test) > 0 else 0.0
        returns_mean = float(np.mean(all_returns_test)) if len(all_returns_test) > 0 else 0.0
        collision_mean = float(np.mean(all_collision_rates_test)) if len(all_collision_rates_test) > 0 else 0.0
        final_goal_dist_mean = float(np.mean([d for d in final_goal_dists if d is not None])) if any(d is not None for d in final_goal_dists) else None
        obstacle_top_z_mean = float(np.mean(obstacle_top_zs)) if len(obstacle_top_zs) > 0 else None
        self.logger_util_test.record(sr_mean)

        log_data = {
            "mean_success_rates_test": sr_mean,
            "mean_returns_test": returns_mean,
            "mean_collision_rate_test": collision_mean,
            "SR_test_L3": self.logger_util_test.average_of_largest_K(),
            "test_mean_score": sr_mean,
        }

        if final_goal_dist_mean is not None:
            log_data["mean_final_goal_dist_test"] = final_goal_dist_mean
        if obstacle_top_z_mean is not None:
            log_data["mean_obstacle_top_z_test"] = obstacle_top_z_mean

        if len(saved_local_videos) > 0:
            cprint(f"Saved {len(saved_local_videos)} rollout videos to {os.path.join(self.output_dir, self.local_video_dir)}", "cyan")
            log_data["local_rollout_video_dir_test"] = os.path.join(self.output_dir, self.local_video_dir)
            log_data["local_rollout_video_count_test"] = int(len(saved_local_videos))

        if self.log_wandb_videos and len(cam0_first_episode) > 0:
            cam0 = np.stack(cam0_first_episode, axis=0)
            cam1 = np.stack(cam1_first_episode, axis=0)
            merged = np.concatenate([cam0, cam1], axis=2).transpose(0, 3, 1, 2)
            log_data["sim_video_cam01_merged_test"] = wandb.Video(
                merged.astype(np.uint8), fps=self.fps, format="mp4"
            )
            try:
                videos_test = self.env_test.env.get_video()
                log_data["sim_video_test"] = wandb.Video(
                    videos_test.astype(np.uint8), fps=self.fps, format="mp4"
                )
            except Exception as exc:
                cprint(f"Motion-plan video capture failed: {exc}", "yellow")

        return log_data


class Lite6PickPlaceRunner(Lite6BaseRunner):
    def __init__(
        self,
        output_dir,
        n_train=10,
        n_test=1,
        max_steps=350,
        n_obs_steps=2,
        n_action_steps=8,
        fps=10,
        use_gui=False,
        num_points=2500,
        image_size=224,
        action_dim=7,
        max_steps_per_waypoint=4,
        waypoint_threshold=0.02,
        capture_table=False,
        use_workspace_crop=True,
        workspace_std=30.0,
        log_wandb_videos=True,
        save_local_videos=False,
        save_all_local_episodes=False,
        merge_cams_side_by_side=True,
        local_video_dir="rollout_videos",
    ):
        super().__init__(output_dir)
        self.episode_test = n_test
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.action_dim = action_dim
        self.num_points = num_points
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.log_wandb_videos = log_wandb_videos
        self.save_local_videos = save_local_videos
        self.save_all_local_episodes = save_all_local_episodes
        self.merge_cams_side_by_side = merge_cams_side_by_side
        self.local_video_dir = local_video_dir
        self.include_gripper = (action_dim == 7 or action_dim == 13)

        from diffusion_policy_3d.env.pybullet.pybullet_wrapper import Lite6PickPlaceEnv

        def env_fn():
             return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    Lite6PickPlaceEnv(
                        use_gui=use_gui,
                        num_points=num_points,
                        image_size=image_size,
                        action_dim=action_dim,
                        max_steps=max_steps,
                        max_steps_per_waypoint=max_steps_per_waypoint,
                        waypoint_threshold=waypoint_threshold,
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method="sum",
            )
        
        self.env_test = env_fn()

    def run(self, policy: BasePolicy, dataset=None):
        device = policy.device
        all_returns_test = []
        all_success_rates_test = []
        final_goal_dists = []
        saved_local_videos = []
        wandb_video = None

        base_env = self.env_test.env.env
        val_episode_indices = None
        if dataset is not None and hasattr(dataset, "train_mask"):
            val_episode_indices = np.where(~dataset.train_mask)[0]

        for episode_id in range(self.episode_test):
            cube_start_pos = None
            cube_start_orn = None
            if dataset is not None and hasattr(dataset, "get_episode_cube_start_pos"):
                try:
                    if val_episode_indices is not None and episode_id < len(val_episode_indices):
                        dataset_episode_idx = int(val_episode_indices[episode_id])
                    else:
                        dataset_episode_idx = episode_id
                    if dataset_episode_idx < dataset.replay_buffer.n_episodes:
                        cube_info = dataset.get_episode_cube_start_pos(dataset_episode_idx)
                        cube_start_pos = cube_info[:3].tolist()
                        cube_start_orn = cube_info[3:7].tolist()
                except Exception as exc:
                    cprint(f"PickPlace eval falling back to default cube pose: {exc}", "yellow")
                    cube_start_pos = None
                    cube_start_orn = None

            obs = self.env_test.reset(
                cube_start_pos=cube_start_pos,
                cube_start_orn=cube_start_orn,
            )
            policy.reset()

            reward_sum = 0.0
            done = False
            final_goal_dist = None

            while not done:
                policy_obs = self._to_policy_obs(obs, policy, device)
                with torch.no_grad():
                    action_dict = policy.predict_action(policy_obs)
                action = dict_apply(action_dict, lambda x: x.detach().cpu().numpy())["action"]

                if action.ndim == 3:
                    action = action.squeeze(0)
                elif action.ndim == 1:
                    action = action[None, :]

                action_seq = action[: self.n_action_steps]
                obs, reward, done, info = self.env_test.step(action_seq)
                reward_sum += float(reward)
                done = bool(np.all(done))

                if base_env.cube_id is not None:
                    import pybullet as p
                    cube_pos, _ = p.getBasePositionAndOrientation(base_env.cube_id)
                    final_goal_dist = float(
                        np.linalg.norm(np.array(cube_pos[:2]) - np.array([0.30, 0.20]))
                    )

            success = 1.0 if base_env.is_success() else 0.0
            all_returns_test.append(reward_sum)
            all_success_rates_test.append(success)
            final_goal_dists.append(final_goal_dist)

            if self.log_wandb_videos and episode_id == 0:
                try:
                    wandb_video = self.env_test.env.get_video()
                except Exception as exc:
                    cprint(f"PickPlace video capture failed: {exc}", "yellow")

            if self.save_local_videos:
                try:
                    frames = self.env_test.env.get_video().transpose(0, 2, 3, 1)
                    frames = self._annotate_frames_with_final_error(list(frames), final_goal_dist)
                    out_dir = os.path.join(self.output_dir, self.local_video_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"pick_place_episode_{episode_id:03d}.mp4")
                    if self._write_rgb_video(frames, out_path):
                        saved_local_videos.append(out_path)
                except Exception as exc:
                    cprint(f"PickPlace local video save failed: {exc}", "yellow")

            cprint(
                f"Lite6 PickPlace Episode {episode_id}: Reward={reward_sum:.2f}, "
                f"Success={bool(success)}, Final XY error={final_goal_dist}",
                "yellow",
            )

        sr_mean = float(np.mean(all_success_rates_test)) if all_success_rates_test else 0.0
        returns_mean = float(np.mean(all_returns_test)) if all_returns_test else 0.0
        final_goal_dist_mean = (
            float(np.mean([d for d in final_goal_dists if d is not None]))
            if any(d is not None for d in final_goal_dists)
            else None
        )
        self.logger_util_test.record(sr_mean)

        log_data = {
            "mean_success_rates_test": sr_mean,
            "mean_returns_test": returns_mean,
            "SR_test_L3": self.logger_util_test.average_of_largest_K(),
            "test_mean_score": sr_mean,
        }
        if final_goal_dist_mean is not None:
            log_data["mean_final_goal_dist_test"] = final_goal_dist_mean
        if len(saved_local_videos) > 0:
            log_data["local_rollout_video_dir_test"] = os.path.join(self.output_dir, self.local_video_dir)
            log_data["local_rollout_video_count_test"] = int(len(saved_local_videos))
        if wandb_video is not None:
            log_data["sim_video_test"] = wandb.Video(
                wandb_video.astype(np.uint8), fps=self.fps, format="mp4"
            )

        return log_data
