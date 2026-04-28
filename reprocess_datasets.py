import os
import glob
import argparse
import numpy as np
import pybullet as p
import fpsample
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

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

def farthest_point_sampling(points, n_samples):
    if len(points) <= n_samples:
        return points
    indices = fpsample.fps_npdu_sampling(points, n_samples)
    return points[indices]

def sample_or_pad(points, num_target):
    if len(points) >= num_target:
        return farthest_point_sampling(points, num_target)
    elif len(points) > 0:
        pad_indices = np.random.choice(len(points), size=(num_target - len(points)), replace=True)
        return np.vstack([points, points[pad_indices]])
    else:
        return np.zeros((num_target, 3))

def process_single_frame(depth_path, seg_path, out_path, view_matrix, proj_matrix, base_pos, robot_id=2):
    """Processes a single frame. Safe for multiprocessing."""
    if os.path.exists(out_path):
        return  # Skip already processed frames
    
    try:
        depth_buffer = np.load(depth_path)
        seg_tp = np.load(seg_path)

        pcd_base = depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix, base_pos)
        pts_flat = pcd_base.reshape(-1, 3)

        seg_body_ids = seg_tp.flatten() & 0xFFFFFF

        # Create boolean masks
        robot_mask = (seg_body_ids == robot_id)
        # Objects are anything that is not robot, and not the plane(0) / table(1)
        obj_mask = (seg_body_ids != robot_id) & (seg_body_ids != 0) & (seg_body_ids != 1)

        valid_mask = (pts_flat[:, 2] < 2.5) & ~(depth_buffer >= 0.9999).flatten()

        robot_pts = pts_flat[robot_mask & valid_mask]
        obj_pts = pts_flat[obj_mask & valid_mask]

        sampled_rob = sample_or_pad(robot_pts, 2000)
        sampled_obj = sample_or_pad(obj_pts, 500)

        final_pcd = np.concatenate([sampled_rob, sampled_obj], axis=0)

        # Shuffle points to avoid ordering bias for neural networks
        shuffle_idx = np.random.permutation(len(final_pcd))
        final_pcd = final_pcd[shuffle_idx]

        assert final_pcd.shape == (2500, 3)
        np.save(out_path, final_pcd)

    except Exception as e:
        print(f"Error processing {depth_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch reprocess simulated depth to class-aware partitioned PCDs.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing iter_* folders")
    args = parser.parse_args()

    # Pre-compute Projection & View globally (Same across all simulations)
    p.connect(p.DIRECT)
    eye = np.array([0.7463, 0.3093, 1.1774])
    direction = np.array([-0.7751, -0.4045, -0.4855])
    direction = direction / np.linalg.norm(direction)
    tp_cam_eye = (eye - 0.2 * direction).tolist()
    tp_cam_target = (tp_cam_eye + 1.0 * direction).tolist()
    tp_cam_up = [0, 0, 1]

    view_matrix = p.computeViewMatrix(tp_cam_eye, tp_cam_target, tp_cam_up)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=3.0)
    base_pos = [0, 0, 0.62]
    p.disconnect() # No longer needed

    # Collect all tasks
    tasks = []
    
    iter_folders = sorted(glob.glob(os.path.join(args.data_dir, "iter_*")))
    if not iter_folders:
        print(f"No iter_ folders found in {args.data_dir}")
        return

    print(f"Found {len(iter_folders)} iteration folders. Resolving file paths...")

    for iter_dir in iter_folders:
        # Dynamically find any camera folders inside the iter dir
        # A camera folder is valid if it contains 'depth' and 'segmentation' subdirs
        subdirs = next(os.walk(iter_dir))[1]
        for sub in subdirs:
            cam_dir = os.path.join(iter_dir, sub)
            depth_dir = os.path.join(cam_dir, "depth")
            seg_dir = os.path.join(cam_dir, "segmentation")

            if os.path.isdir(depth_dir) and os.path.isdir(seg_dir):
                out_dir = os.path.join(cam_dir, "seg_pc")
                os.makedirs(out_dir, exist_ok=True)

                depth_files = glob.glob(os.path.join(depth_dir, "*.npy"))
                for depth_path in depth_files:
                    filename = os.path.basename(depth_path)
                    
                    # Deduce segmentation equivalent filename
                    # usually tp_depth_0000.npy -> tp_seg_0000.npy
                    seg_name = filename.replace("_depth_", "_seg_")
                    seg_path = os.path.join(seg_dir, seg_name)

                    if os.path.exists(seg_path):
                        out_name = filename.replace("_depth_", "_pcd_")
                        out_path = os.path.join(out_dir, out_name)
                        tasks.append((depth_path, seg_path, out_path))

    print(f"Discovered {len(tasks)} frames to process.")
    
    if len(tasks) > 0:
        # Use multiprocessing to speed up the batch processing
        num_workers = min(multiprocessing.cpu_count(), 16)
        print(f"Processing in parallel using {num_workers} workers...")
        
        # Track progress simply
        completed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_single_frame, d_path, s_path, o_path, view_matrix, proj_matrix, base_pos)
                for d_path, s_path, o_path in tasks
            ]
            
            for future in futures:
                future.result() # Will block until completion
                completed += 1
                if completed % 1000 == 0:
                    print(f"  Processed {completed} / {len(tasks)} frames...")

        print("Dataset processing complete!")

if __name__ == '__main__':
    main()
