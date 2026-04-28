#!/usr/bin/env python3
import argparse
import csv
import glob
import os

import numpy as np
import pybullet as p


def normalize_yaw(yaw_rad):
    """Normalize box yaw to the [-pi/2, pi/2] equivalent range."""
    return ((yaw_rad + np.pi / 2) % np.pi) - np.pi / 2


def estimate_yaw_from_points(points):
    """Estimate obstacle yaw from visible XY points using PCA.

    The collected obstacle has a nearly square footprint, so this is an
    observation-derived estimate, not the exact sampled yaw. The returned
    ratio is larger when one principal XY direction is clearly dominant.
    """
    xy = np.asarray(points[:, :2], dtype=np.float64)
    if xy.shape[0] < 3:
        return None, None

    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = np.cov(xy, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvec = eigvecs[:, order[0]]

    yaw = float(np.arctan2(eigvec[1], eigvec[0]))
    yaw = float(normalize_yaw(yaw))
    ratio = float(eigvals[0] / max(eigvals[1], 1e-12))
    return yaw, ratio


def depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix, base_pos, width=224, height=224):
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
    return points_world - np.array(base_pos)


def get_camera_matrices():
    p.connect(p.DIRECT)
    eye = np.array([0.7463, 0.3093, 1.1774])
    direction = np.array([-0.7751, -0.4045, -0.4855])
    direction = direction / np.linalg.norm(direction)
    cam_eye = (eye - 0.2 * direction).tolist()
    cam_target = (cam_eye + 1.0 * direction).tolist()
    cam_up = [0, 0, 1]

    view_matrix = p.computeViewMatrix(cam_eye, cam_target, cam_up)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=3.0)
    p.disconnect()
    return view_matrix, proj_matrix


def paired_frame_paths(iter_dir, camera):
    depth_dir = os.path.join(iter_dir, camera, "depth")
    seg_dir = os.path.join(iter_dir, camera, "segmentation")
    depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    for depth_path in depth_paths:
        seg_name = os.path.basename(depth_path).replace("_depth_", "_seg_")
        seg_path = os.path.join(seg_dir, seg_name)
        if os.path.exists(seg_path):
            yield depth_path, seg_path


def frame_obstacle_stats(
    depth_path,
    seg_path,
    view_matrix,
    proj_matrix,
    base_pos,
    robot_id,
    excluded_body_ids,
    table_world_z,
    image_size,
):
    depth = np.load(depth_path)
    seg = np.load(seg_path)
    points = depth_to_point_cloud(
        depth,
        view_matrix,
        proj_matrix,
        base_pos,
        width=image_size,
        height=image_size,
    )

    body_ids = seg.flatten() & 0xFFFFFF
    valid_mask = (depth.flatten() < 0.9999) & (points[:, 2] < 2.5)
    excluded = np.isin(body_ids, np.asarray(excluded_body_ids, dtype=np.int64))
    obstacle_mask = valid_mask & (~excluded) & (body_ids != robot_id)
    obstacle_points = points[obstacle_mask]

    if len(obstacle_points) == 0:
        return None

    z_min_base = float(obstacle_points[:, 2].min())
    z_max_base = float(obstacle_points[:, 2].max())
    z_max_world = z_max_base + float(base_pos[2])
    yaw_rad, yaw_confidence = estimate_yaw_from_points(obstacle_points)
    return {
        "frame": os.path.basename(depth_path),
        "num_points": int(len(obstacle_points)),
        "z_min_base": z_min_base,
        "z_max_base": z_max_base,
        "visible_height": float(z_max_base - z_min_base),
        "z_max_world": z_max_world,
        "height_above_table": float(z_max_world - table_world_z),
        "yaw_rad_est": yaw_rad,
        "yaw_deg_est": None if yaw_rad is None else float(np.degrees(yaw_rad)),
        "yaw_confidence_ratio": yaw_confidence,
    }


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate motion-planning obstacle max z from raw depth and segmentation masks."
    )
    parser.add_argument("--data-dir", default="/scratch3/cross-emb/dataset_mp_poco")
    parser.add_argument("--camera", default="third_person")
    parser.add_argument("--output-csv", default="motion_obstacle_heights.csv")
    parser.add_argument("--robot-id", type=int, default=2)
    parser.add_argument("--exclude-body-ids", type=int, nargs="*", default=[0, 1, -1])
    parser.add_argument("--base-pos", type=float, nargs=3, default=[0.0, 0.0, 0.62])
    parser.add_argument("--table-world-z", type=float, default=0.625)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    view_matrix, proj_matrix = get_camera_matrices()
    iter_dirs = sorted(glob.glob(os.path.join(args.data_dir, "iter_*")))
    if args.max_episodes is not None:
        iter_dirs = iter_dirs[: args.max_episodes]

    rows = []
    episode_rows = []
    for episode_idx, iter_dir in enumerate(iter_dirs):
        frame_stats = []
        for depth_path, seg_path in paired_frame_paths(iter_dir, args.camera):
            stats = frame_obstacle_stats(
                depth_path=depth_path,
                seg_path=seg_path,
                view_matrix=view_matrix,
                proj_matrix=proj_matrix,
                base_pos=args.base_pos,
                robot_id=args.robot_id,
                excluded_body_ids=args.exclude_body_ids,
                table_world_z=args.table_world_z,
                image_size=args.image_size,
            )
            if stats is None:
                continue
            stats["episode"] = os.path.basename(iter_dir)
            rows.append(stats)
            frame_stats.append(stats)

        if frame_stats:
            best = max(frame_stats, key=lambda row: row["z_max_world"])
            episode_rows.append(best)

        if (episode_idx + 1) % 25 == 0:
            print(f"Processed {episode_idx + 1}/{len(iter_dirs)} episodes")

    if not rows:
        raise RuntimeError(f"No obstacle points found under {args.data_dir}")

    fieldnames = [
        "episode",
        "frame",
        "num_points",
        "z_min_base",
        "z_max_base",
        "visible_height",
        "z_max_world",
        "height_above_table",
        "yaw_rad_est",
        "yaw_deg_est",
        "yaw_confidence_ratio",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    frame_summary = summarize([row["z_max_world"] for row in rows])
    episode_summary = summarize([row["z_max_world"] for row in episode_rows])
    yaw_rows = [row for row in episode_rows if row["yaw_deg_est"] is not None]

    print(f"Wrote per-frame obstacle stats to {args.output_csv}")
    print("Frame z_max_world summary:")
    for key, value in frame_summary.items():
        print(f"  {key}: {value}")
    print("Episode max z_max_world summary:")
    for key, value in episode_summary.items():
        print(f"  {key}: {value}")
    if yaw_rows:
        print("Episode yaw_deg_est summary from max-height frame:")
        for key, value in summarize([row["yaw_deg_est"] for row in yaw_rows]).items():
            print(f"  {key}: {value}")
        print("Episode yaw_confidence_ratio summary:")
        for key, value in summarize([row["yaw_confidence_ratio"] for row in yaw_rows]).items():
            print(f"  {key}: {value}")
        print("Note: yaw is estimated from visible obstacle points; low confidence ratios mean the footprint is close to square or partially occluded.")


if __name__ == "__main__":
    main()
