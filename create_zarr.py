#!/usr/bin/env python3
"""
Convert simulation pick-and-place dataset into a Zarr store.

Directory layout expected (relative to DATA_ROOT):
  dataset/
    iter_0000/
      agent_pos.npy          # (N, 7) — 6 arm joints + 1 normalised gripper
      actions.npy            # (N, 7) — absolute next-state targets
      cube_pos.npy           # (N, 7) — xyz + quaternion
      third_person/
        rgb/   tp_rgb_XXXX.png
        pcd/   tp_pcd_XXXX.npy   (2500, 3)
    iter_0001/
      ...

Actions stored in the Zarr:
  - First 6 dims (joints): kept as absolute targets  (actions[:, :6])
  - 7th dim (gripper):     stored as delta = action_gripper - state_gripper
"""

import os
import shutil
import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = "/scratch3/cross-emb/dataset_pick_place_poco"
OUT_ZARR  = "/scratch3/cross-emb/dataset_pick_place_poco_2.zarr"

CHUNK_SIZE = 100  # samples per Zarr chunk

# ── accumulators ─────────────────────────────────────────────────────────────
all_imgs    = []
all_pcs     = []
all_actions = []
all_states  = []
episode_ends = []
current_idx  = 0

# ── discover trajectories ───────────────────────────────────────────────────
trajectories = sorted(
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("iter_")
)
print(f"Found {len(trajectories)} trajectories in {DATA_ROOT}")

for traj in tqdm(trajectories, desc="Trajectories"):
    traj_dir = os.path.join(DATA_ROOT, traj)

    # ── load states & actions (.npy) ─────────────────────────────────────
    state_path  = os.path.join(traj_dir, "agent_pos.npy")
    action_path = os.path.join(traj_dir, "actions.npy")

    if not os.path.exists(state_path) or not os.path.exists(action_path):
        print(f"  Skipping {traj}: missing agent_pos.npy or actions.npy")
        continue

    states  = np.load(state_path)   # (N, 7)
    actions = np.load(action_path)  # (N, 7)

    if states.ndim == 1:
        states = states.reshape(1, -1)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)

    num_frames = len(states)

    # ── gripper delta: action[i] - state[i] for dim 6 ───────────────────
    actions[:, 6] = actions[:, 6] - states[:, 6]

    # ── RGB images ───────────────────────────────────────────────────────
    rgb_dir = os.path.join(traj_dir, "third_person", "rgb")
    rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))

    if len(rgb_files) != num_frames:
        print(f"  Skipping {traj}: {num_frames} states vs {len(rgb_files)} images")
        continue

    for f in rgb_files:
        img = np.array(Image.open(os.path.join(rgb_dir, f)))
        all_imgs.append(img)

    # ── Point clouds (.npy) ──────────────────────────────────────────────
    pcd_dir = os.path.join(traj_dir, "third_person", "seg_pc")
    pcd_files = sorted(f for f in os.listdir(pcd_dir) if f.endswith(".npy"))

    if len(pcd_files) != num_frames:
        print(f"  Skipping {traj}: {num_frames} states vs {len(pcd_files)} point clouds")
        # undo images we just appended
        for _ in range(len(rgb_files)):
            all_imgs.pop()
        continue

    for f in pcd_files:
        pc = np.load(os.path.join(pcd_dir, f))  # (2500, 3)
        all_pcs.append(pc)

    # ── accumulate ───────────────────────────────────────────────────────
    all_states.append(states)
    all_actions.append(actions)

    current_idx += num_frames
    episode_ends.append(current_idx)

# ── stack everything ─────────────────────────────────────────────────────────
print("\nStacking arrays …")
all_imgs     = np.stack(all_imgs, axis=0)           # (total, H, W, 3)
all_pcs      = np.stack(all_pcs, axis=0)            # (total, 2500, 3)
all_actions  = np.vstack(all_actions)               # (total, 7)
all_states   = np.vstack(all_states)                # (total, 7)
episode_ends = np.array(episode_ends, dtype=np.int64)

print(f"  images       : {all_imgs.shape}")
print(f"  point_clouds : {all_pcs.shape}")
print(f"  actions      : {all_actions.shape}  (joints absolute, gripper delta)")
print(f"  states       : {all_states.shape}")
print(f"  episodes     : {len(episode_ends)}")

# ── write Zarr ───────────────────────────────────────────────────────────────
if os.path.exists(OUT_ZARR):
    print(f"Removing existing {OUT_ZARR}")
    shutil.rmtree(OUT_ZARR)

zroot = zarr.open(OUT_ZARR, mode="w")
compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

data = zroot.create_group("data")
meta = zroot.create_group("meta")

data.create_dataset("img",         data=all_imgs,    chunks=(CHUNK_SIZE, *all_imgs.shape[1:]),   dtype="uint8",   compressor=compressor)
data.create_dataset("point_cloud", data=all_pcs,     chunks=(CHUNK_SIZE, *all_pcs.shape[1:]),    dtype="float32", compressor=compressor)
data.create_dataset("action",      data=all_actions,  chunks=(CHUNK_SIZE, all_actions.shape[1]),  dtype="float32", compressor=compressor)
data.create_dataset("state",       data=all_states,   chunks=(CHUNK_SIZE, all_states.shape[1]),   dtype="float32", compressor=compressor)
meta.create_dataset("episode_ends", data=episode_ends, chunks=(len(episode_ends),),               dtype="int64",   compressor=compressor)

print(f"\n✅ Saved Zarr to {OUT_ZARR}")
print("Structure:")
print(f"  data/img          {all_imgs.shape}")
print(f"  data/point_cloud  {all_pcs.shape}")
print(f"  data/action       {all_actions.shape}")
print(f"  data/state        {all_states.shape}")
print(f"  meta/episode_ends {episode_ends.shape}")
