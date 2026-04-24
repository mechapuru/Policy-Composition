#!/usr/bin/env python3
"""
Convert multi-view motion-planning simulation dataset into a Zarr store.

Expected directory layout (relative to DATA_ROOT):
  dataset_mp/
    iter_0000/
      agent_pos.npy
      actions.npy
      camera_poses_summary.json
      thirdperson_cam00/
        rgb/  thirdperson_cam00_rgb_XXXX.png
        pcd/  thirdperson_cam00_pcd_XXXX.npy
      thirdperson_cam01/
        rgb/  thirdperson_cam01_rgb_XXXX.png
        pcd/  thirdperson_cam01_pcd_XXXX.npy
      thirdperson_cam02/
        rgb/  thirdperson_cam02_rgb_XXXX.png
        pcd/  thirdperson_cam02_pcd_XXXX.npy

Action convention in output:
  - joints [0:6): absolute targets
  - gripper [6]:  delta = action_gripper - state_gripper at the same time-step

Alignment convention in output:
  - keep all state/image/point-cloud frames
    - actions and states are required to have equal length (strict)

Point cloud convention in output:
  - each frame point cloud is merged by concatenating cam00 -> cam01 -> cam02
  - all input clouds are assumed to already be in the robot base frame
"""

import json
import os
import shutil
from typing import Dict, List, Optional
import fpsample
import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm

# CONFIG
DATA_ROOT = "/scratch3/cross-emb/dataset_mp_poco"
OUT_ZARR = "/scratch3/cross-emb/dataset_mp_poco_2.zarr"
CHUNK_SIZE = 100
# Set to True for the 3-camera layout produced by motion_plan_data().
# Set to False for the single-camera layout produced by the grab-cube collector.
USE_3CAM = False
CAMERAS = (
    ("thirdperson_cam00", "thirdperson_cam01", "thirdperson_cam02")
    if USE_3CAM
    else ("tp",)
)
TARGET_MERGED_POINTS = 5000
# Set to None to include all episodes, or a float (radians) to keep only episodes <= threshold.
TRACKING_ERROR_MAX_RAD = None

# OS UTILS
def sorted_files(directory: str, extension: str) -> List[str]:
    return sorted(f for f in os.listdir(directory) if f.endswith(extension))

def load_camera_pose_summary(traj_dir: str) -> Optional[dict]:
    summary_path = os.path.join(traj_dir, "camera_poses_summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)
def load_configuration_vector(traj_dir: str, filename: str, traj_name: str) -> Optional[np.ndarray]:
    """Load start/end configuration JSON and flatten to a fixed 14D vector."""
    path = os.path.join(traj_dir, filename)
    if not os.path.exists(path):
        print(f"  Skipping {traj_name}: missing {filename}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    try:
        joint_positions = np.asarray(cfg["joint_positions"], dtype=np.float32).reshape(-1)
        eef_position = np.asarray(cfg["eef_position"], dtype=np.float32).reshape(-1)
        eef_orientation = np.asarray(cfg["eef_orientation"], dtype=np.float32).reshape(-1)
        gripper_state = np.asarray([cfg["gripper_state"]], dtype=np.float32)
    except KeyError as exc:
        print(f"  Skipping {traj_name}: missing key {exc} in {filename}")
        return None

    if len(joint_positions) != 6 or len(eef_position) != 3 or len(eef_orientation) != 4:
        print(
            f"  Skipping {traj_name}: invalid shape in {filename} "
            f"(joint={len(joint_positions)}, eef_pos={len(eef_position)}, eef_orn={len(eef_orientation)})"
        )
        return None

    return np.concatenate([joint_positions, eef_position, eef_orientation, gripper_state], axis=0)


def load_tracking_error_rad(traj_dir: str, traj_name: str) -> Optional[float]:
    """Load tracking_err_rad from state_action.json. Returns None if missing."""
    path = os.path.join(traj_dir, "state_action.json")
    if not os.path.exists(path):
        print(f"  {traj_name}: state_action.json missing; tracking_err_rad unavailable")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    value = data.get("tracking_err_rad", None)
    if value is None:
        print(f"  {traj_name}: tracking_err_rad missing in state_action.json")
        return None
    return float(value)
# FPS UTILS 
def farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Farthest Point Sampling using fpsample (Rust-backed)."""
    if len(points) <= n_samples:
        return points
    indices = fpsample.fps_npdu_sampling(points, n_samples)
    return points[indices]

def normalize_merged_point_cloud(merged_pc: np.ndarray, target_points: int) -> np.ndarray:
    """
    Return merged cloud with exactly target_points rows.

    - If more points: downsample with farthest point sampling.
    - If fewer points: tile points and fill remainder deterministically.
    - If empty: fill with zeros.
    """
    n = merged_pc.shape[0]
    if n == target_points:
        return merged_pc
    if n == 0:
        return np.zeros((target_points, 3), dtype=np.float32)
    if n > target_points:
        return farthest_point_sampling(merged_pc, target_points).astype(np.float32, copy=False)

    reps = target_points // n
    rem = target_points % n
    out = np.tile(merged_pc, (reps, 1))
    if rem > 0:
        out = np.concatenate([out, merged_pc[:rem]], axis=0)
    return out.astype(np.float32, copy=False)

# MAIN 
def process_episode(traj_dir: str, traj_name: str) -> Optional[dict]:
    """
    Process one trajectory and return episode payload.
    Returns None when an episode is invalid and should be skipped.
    """
    state_path = os.path.join(traj_dir, "agent_pos.npy")
    action_path = os.path.join(traj_dir, "actions.npy")
    if not os.path.exists(state_path) or not os.path.exists(action_path):
        print(f"  Skipping {traj_name}: missing agent_pos.npy or actions.npy")
        return None

    states = np.load(state_path)
    actions = np.load(action_path)

    start_cfg = load_configuration_vector(traj_dir, "start_configuration.json", traj_name)
    end_cfg = load_configuration_vector(traj_dir, "end_configuration.json", traj_name)
    if start_cfg is None or end_cfg is None:
        return None

    tracking_err_rad = load_tracking_error_rad(traj_dir, traj_name)
    if TRACKING_ERROR_MAX_RAD is not None:
        if tracking_err_rad is None:
            print(f"  Skipping {traj_name}: tracking_err_rad required by threshold filter")
            return None
        if tracking_err_rad > float(TRACKING_ERROR_MAX_RAD):
            print(
                f"  Skipping {traj_name}: tracking_err_rad={tracking_err_rad:.6f} "
                f"> {TRACKING_ERROR_MAX_RAD}"
            )
            return None

    if states.ndim == 1:
        states = states.reshape(1, -1)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)

    if states.shape[1] != 7 or actions.shape[1] != 7:
        print(f"  Skipping {traj_name}: expected state/action dim 7, got {states.shape} and {actions.shape}")
        return None

    num_states = states.shape[0]
    num_actions = actions.shape[0]

    if num_actions != num_states:
        raise RuntimeError(
            f"{traj_name}: actions and states must have equal length, got "
            f"actions={num_actions}, states={num_states}"
        )
    # Convert gripper channel to delta at the same time-step.
    actions_delta = actions.astype(np.float32, copy=True)
    if num_actions > 0:
        actions_delta[:, 6] = actions_delta[:, 6] - states[:, 6]

    rgb_files_by_cam: Dict[str, List[str]] = {}
    pcd_files_by_cam: Dict[str, List[str]] = {}

    if USE_3CAM:
        for cam in CAMERAS:
            rgb_dir = os.path.join(traj_dir, cam, "rgb")
            pcd_dir = os.path.join(traj_dir, cam, "seg_pc")
            if not os.path.isdir(rgb_dir) or not os.path.isdir(pcd_dir):
                print(f"  Skipping {traj_name}: missing rgb/pcd folder for {cam}")
                return None

            rgb_files = sorted_files(rgb_dir, ".png")
            pcd_files = sorted_files(pcd_dir, ".npy")

            if len(rgb_files) != num_states:
                print(f"  Skipping {traj_name}: {cam} rgb count {len(rgb_files)} != states {num_states}")
                return None
            if len(pcd_files) != num_states:
                print(f"  Skipping {traj_name}: {cam} pcd count {len(pcd_files)} != states {num_states}")
                return None

            rgb_files_by_cam[cam] = rgb_files
            pcd_files_by_cam[cam] = pcd_files
    else:
        cam = CAMERAS[0]
        rgb_dir = os.path.join(traj_dir, "third_person", "rgb")
        pcd_dir = os.path.join(traj_dir, "third_person", "pcd")
        if not os.path.isdir(rgb_dir) or not os.path.isdir(pcd_dir):
            print(f"  Skipping {traj_name}: missing third_person/rgb or third_person/pcd")
            return None

        rgb_files = sorted_files(rgb_dir, ".png")
        pcd_files = sorted_files(pcd_dir, ".npy")

        if len(rgb_files) != num_states:
            print(f"  Skipping {traj_name}: rgb count {len(rgb_files)} != states {num_states}")
            return None
        if len(pcd_files) != num_states:
            print(f"  Skipping {traj_name}: pcd count {len(pcd_files)} != states {num_states}")
            return None

        rgb_files_by_cam[cam] = rgb_files
        pcd_files_by_cam[cam] = pcd_files

    local_imgs: Dict[str, List[np.ndarray]] = {cam: [] for cam in CAMERAS}
    local_pcs: List[np.ndarray] = []

    for i in range(num_states):
        frame_imgs = []
        frame_pcs = []

        for cam in CAMERAS:
            if USE_3CAM:
                rgb_path = os.path.join(traj_dir, cam, "rgb", rgb_files_by_cam[cam][i])
                pcd_path = os.path.join(traj_dir, cam, "pcd", pcd_files_by_cam[cam][i])
            else:
                rgb_path = os.path.join(traj_dir, "third_person", "rgb", rgb_files_by_cam[cam][i])
                pcd_path = os.path.join(traj_dir, "third_person", "pcd", pcd_files_by_cam[cam][i])

            img = np.array(Image.open(rgb_path), dtype=np.uint8)
            pc = np.load(pcd_path).astype(np.float32, copy=False)

            if pc.ndim != 2 or pc.shape[1] != 3:
                print(f"  Skipping {traj_name}: invalid point cloud shape {pc.shape} in {pcd_path}")
                return None

            local_imgs[cam].append(img)
            frame_imgs.append(img)
            frame_pcs.append(pc)

        if USE_3CAM:
            merged_pc = np.concatenate(frame_pcs, axis=0).astype(np.float32, copy=False)
            merged_pc = normalize_merged_point_cloud(merged_pc, TARGET_MERGED_POINTS)
        else:
            merged_pc = frame_pcs[0].astype(np.float32, copy=False)
            TARGET_MERGED_POINTS = merged_pc.shape[0]

        local_pcs.append(merged_pc)

    pose_summary = load_camera_pose_summary(traj_dir)
    print(f"  {traj_name}: synchronized frames verified and  merged+FPS normalized to {TARGET_MERGED_POINTS}")
    return {
        "states": states.astype(np.float32, copy=False),
        "actions": actions_delta.astype(np.float32, copy=False),
        "start_config": start_cfg.astype(np.float32, copy=False),
        "end_config": end_cfg.astype(np.float32, copy=False),
        "tracking_err_rad": np.nan if tracking_err_rad is None else np.float32(tracking_err_rad),
        "imgs": local_imgs,
        "pcs": local_pcs,
        "pose_summary": pose_summary,
        "num_frames": num_states,
    }


def main() -> None:
    trajectories = sorted(
        d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("iter_")
    )
    print(f"Found {len(trajectories)} trajectories in {DATA_ROOT}")

    all_imgs: Dict[str, List[np.ndarray]] = {cam: [] for cam in CAMERAS}
    all_pcs: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_states: List[np.ndarray] = []
    all_start_configs: List[np.ndarray] = []
    all_end_configs: List[np.ndarray] = []
    all_tracking_err_rad: List[np.float32] = []
    episode_ends: List[int] = []
    episode_names: List[str] = []
    camera_poses_by_episode: Dict[str, dict] = {}

    current_idx = 0

    for traj in tqdm(trajectories, desc="Trajectories"):
        traj_dir = os.path.join(DATA_ROOT, traj)
        payload = process_episode(traj_dir, traj)
        if payload is None:
            continue

        episode_pcs = payload["pcs"]
        if not episode_pcs:
            print(f"  Skipping {traj}: empty point cloud sequence")
            continue

        all_states.append(payload["states"])
        all_actions.append(payload["actions"])
        all_start_configs.append(payload["start_config"])
        all_end_configs.append(payload["end_config"])
        all_tracking_err_rad.append(np.float32(payload["tracking_err_rad"]))
        all_pcs.extend(episode_pcs)

        for cam in CAMERAS:
            all_imgs[cam].extend(payload["imgs"][cam])

        if payload["pose_summary"] is not None:
            camera_poses_by_episode[traj] = payload["pose_summary"]

        num_frames = int(payload["num_frames"])
        current_idx += num_frames
        episode_ends.append(current_idx)
        episode_names.append(traj)

    if not episode_ends:
        raise RuntimeError("No valid episodes found. Nothing to write.")

    # Stack arrays
    print("\nStacking arrays ...")
    imgs_stacked = {cam: np.stack(all_imgs[cam], axis=0).astype(np.uint8, copy=False) for cam in CAMERAS}
    pcs_stacked = np.stack(all_pcs, axis=0).astype(np.float32, copy=False)
    actions_stacked = np.vstack(all_actions).astype(np.float32, copy=False)
    states_stacked = np.vstack(all_states).astype(np.float32, copy=False)
    start_configs_stacked = np.stack(all_start_configs, axis=0).astype(np.float32, copy=False)
    end_configs_stacked = np.stack(all_end_configs, axis=0).astype(np.float32, copy=False)
    tracking_err_rad_arr = np.asarray(all_tracking_err_rad, dtype=np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)
    camera_pose_json_arr = np.asarray(
        [json.dumps(camera_poses_by_episode.get(ep, {})) for ep in episode_names],
        dtype="U",
    )

    # Sort helper: NaNs go last.
    tracking_for_sort = np.where(np.isnan(tracking_err_rad_arr), np.inf, tracking_err_rad_arr)
    tracking_sorted_episode_idx = np.argsort(tracking_for_sort).astype(np.int64)
    if TRACKING_ERROR_MAX_RAD is None:
        tracking_quality_flag = np.ones(len(tracking_err_rad_arr), dtype=np.bool_)
    else:
        tracking_quality_flag = (tracking_err_rad_arr <= float(TRACKING_ERROR_MAX_RAD))

    total_frames = states_stacked.shape[0]
    if (
        actions_stacked.shape[0] != total_frames
        or pcs_stacked.shape[0] != total_frames
        or any(imgs_stacked[cam].shape[0] != total_frames for cam in CAMERAS)
    ):
        raise RuntimeError("Final stacked arrays are misaligned")

    print(f"  frames        : {total_frames}")
    print(f"  state         : {states_stacked.shape}")
    print(f"  action        : {actions_stacked.shape}")
    for cam in CAMERAS:
        print(f"  {cam} rgb     : {imgs_stacked[cam].shape}")
    print(f"  point_cloud   : {pcs_stacked.shape}")
    print(f"  start_config  : {start_configs_stacked.shape}")
    print(f"  end_config    : {end_configs_stacked.shape}")
    print(f"  tracking_err  : {tracking_err_rad_arr.shape}")
    print(f"  episodes      : {len(episode_ends_arr)}")

    # Write Zarr
    if os.path.exists(OUT_ZARR):
        print(f"Removing existing {OUT_ZARR}")
        shutil.rmtree(OUT_ZARR)

    zroot = zarr.open(OUT_ZARR, mode="w")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    data = zroot.create_group("data")
    meta = zroot.create_group("meta")

    for cam in CAMERAS:
        key = cam.replace("thirdperson_", "img_") if USE_3CAM else "img"
        data.create_dataset(key,data=imgs_stacked[cam],chunks=(CHUNK_SIZE, *imgs_stacked[cam].shape[1:]),dtype="uint8",compressor=compressor,)

    data.create_dataset("point_cloud",data=pcs_stacked,chunks=(CHUNK_SIZE, *pcs_stacked.shape[1:]),dtype="float32",compressor=compressor,)
    data.create_dataset("action",data=actions_stacked,chunks=(CHUNK_SIZE, actions_stacked.shape[1]),dtype="float32",compressor=compressor,)
    data.create_dataset("state",data=states_stacked,chunks=(CHUNK_SIZE, states_stacked.shape[1]),dtype="float32",compressor=compressor,)
    data.create_dataset(
        "start_configuration",
        data=start_configs_stacked,
        chunks=(len(start_configs_stacked), start_configs_stacked.shape[1]),
        dtype="float32",
        compressor=compressor,
    )
    data.create_dataset(
        "end_configuration",
        data=end_configs_stacked,
        chunks=(len(end_configs_stacked), end_configs_stacked.shape[1]),
        dtype="float32",
        compressor=compressor,
    )

    meta.create_dataset("episode_ends",data=episode_ends_arr,chunks=(len(episode_ends_arr),),dtype="int64",compressor=compressor,)
    meta.create_dataset(
        "tracking_err_rad",
        data=tracking_err_rad_arr,
        chunks=(len(tracking_err_rad_arr),),
        dtype="float32",
        compressor=compressor,
    )
    meta.create_dataset(
        "tracking_quality_flag",
        data=tracking_quality_flag,
        chunks=(len(tracking_quality_flag),),
        dtype="bool",
        compressor=compressor,
    )
    meta.create_dataset(
        "tracking_err_sorted_episode_idx",
        data=tracking_sorted_episode_idx,
        chunks=(len(tracking_sorted_episode_idx),),
        dtype="int64",
        compressor=compressor,
    )
    meta.create_dataset(
        "camera_poses_summary_json",
        data=camera_pose_json_arr,
        chunks=(len(camera_pose_json_arr),),
        dtype=str,
    )

    # Keep variable-length metadata in attrs (JSON serializable and easy to inspect).
    zroot.attrs["schema_version"] = "motion_planning_multiview_v1"
    zroot.attrs["source_dataset"] = DATA_ROOT
    zroot.attrs["camera_names"] = list(CAMERAS)
    zroot.attrs["use_3cam"] = USE_3CAM
    zroot.attrs["episode_names"] = episode_names
    zroot.attrs["description"] = (
        "Multi-view motion-planning dataset. Three synchronized RGB views are stored "
        "as img_cam00/img_cam01/img_cam02. Point cloud per frame is merged by "
        "concatenating base-frame point clouds in fixed order cam00->cam01->cam02. "
        "Action convention: joints are absolute targets, gripper channel is delta "
        "(action_gripper - state_gripper). Actions and states use strict equal-length alignment."
    )
    zroot.attrs["capture_convention"] = {
        "state": "agent_pos 7D (6 joints + normalized gripper)",
        "action": "7D target with gripper converted to same-step delta",
        "sync": "all modalities synchronized by parsed frame index",
        "start_end_configuration": "[6 joint_positions, 3 eef_position, 4 eef_orientation, 1 gripper_state]",
        "tracking_error": "per-episode tracking_err_rad",
        "point_cloud": (
            "already in base frame; per-frame clouds from all cameras are merged by "
            "camera-order concatenation, then normalized to fixed points using "
            f"farthest point sampling (target={TARGET_MERGED_POINTS})"
        ),
    }
    zroot.attrs["pcd_target_merged_points"] = TARGET_MERGED_POINTS
    zroot.attrs["tracking_error_max_rad_filter"] = TRACKING_ERROR_MAX_RAD

    print(f"\nSaved Zarr to {OUT_ZARR}")
    print("Structure:")
    for cam in CAMERAS:
        key = cam.replace("thirdperson_", "img_")
        print(f"  data/{key:14s} {imgs_stacked[cam].shape}")
    print(f"  data/point_cloud   {pcs_stacked.shape}")
    print(f"  data/action        {actions_stacked.shape}")
    print(f"  data/state         {states_stacked.shape}")
    print(f"  data/start_configuration {start_configs_stacked.shape}")
    print(f"  data/end_configuration   {end_configs_stacked.shape}")
    print(f"  meta/episode_ends  {episode_ends_arr.shape}")
    print(f"  meta/tracking_err_rad {tracking_err_rad_arr.shape}")
    print(f"  meta/camera_poses_summary_json {camera_pose_json_arr.shape}")


if __name__ == "__main__":
    main()
