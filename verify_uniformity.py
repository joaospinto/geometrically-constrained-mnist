import numpy as np
import jax.numpy as jnp
from geometry import DIGIT_SDFS

def gather_pairs(flat_params, group_indices):
    p0 = flat_params[2 * group_indices]
    p1 = flat_params[2 * group_indices + 1]
    return np.stack([p0, p1], axis=-1)

def compute_coverage_ratio(pairs, sdf_fn, grid_size=20, threshold=0.1):
    # Create grid
    ticks = np.linspace(-1.2, 1.2, grid_size)
    X, Y = np.meshgrid(ticks, ticks)
    grid_points = np.stack([X, Y], axis=-1).reshape(-1, 2)
    
    # Grid points inside digit
    is_inside = np.array(sdf_fn(grid_points)) < 0
    inside_points = grid_points[is_inside]
    
    if len(inside_points) == 0:
        return 0.0
    
    # For each inside point, find distance to nearest pair
    # Using broadcasting for small enough grid
    diff = inside_points[:, None, :] - pairs[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    min_dist = np.min(dist, axis=1)
    
    # Ratio of grid cells covered
    covered = np.mean(min_dist < threshold)
    return covered

def main():
    print("Verifying Uniformity...")
    try:
        flat_params = np.load("final_params.npy")
        partition_data = np.load("partition.npz")
    except FileNotFoundError:
        print("Error: Run train.py first.")
        return

    ratios = []
    for k in range(10):
        group_name = f"group_{k}"
        indices = partition_data[group_name]
        pairs = gather_pairs(flat_params, indices)
        
        ratio = compute_coverage_ratio(pairs, DIGIT_SDFS[k])
        print(f"Digit {k} Coverage Ratio: {ratio:.2%}")
        ratios.append(ratio)
    
    print("-" * 20)
    print(f"Mean Coverage Ratio: {np.mean(ratios):.2%}")

if __name__ == "__main__":
    main()
