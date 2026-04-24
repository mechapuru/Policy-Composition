import zarr
import numpy as np

zarr_path = '/scratch3/cross-emb/dataset_pick_place_poco_2.zarr'
root = zarr.open(zarr_path, mode='r')
actions = root['data']['action'][:1000]
print(f"Action shape: {actions.shape}")
print(f"First 10 actions:\n{actions[:10]}")
print(f"Action diffs (first 10):\n{np.diff(actions[:10], axis=0)}")
print(f"Action min: {np.min(actions, axis=0)}")
print(f"Action max: {np.max(actions, axis=0)}")

states = root['data']['state'][:10]
print(f"States (first 10):\n{states}")
