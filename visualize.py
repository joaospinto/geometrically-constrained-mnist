import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

# Import geometry for plotting contours
from geometry import DIGIT_SDFS

def gather_pairs(flat_params, group_indices):
    p0 = flat_params[2 * group_indices]
    p1 = flat_params[2 * group_indices + 1]
    return np.stack([p0, p1], axis=-1)

def main():
    print("Loading data...")
    try:
        flat_params = np.load("final_params.npy")
        partition_data = np.load("partition.npz")
    except FileNotFoundError:
        print("Error: Run train.py first to generate data.")
        return

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # Grid for contours
    grid_res = 100
    x = np.linspace(-1.5, 1.5, grid_res)
    y = np.linspace(-1.5, 1.5, grid_res)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X, Y], axis=-1)
    # Convert to JAX array for SDF
    points_jax = jnp.array(points)

    for k in range(10):
        ax = axes[k]
        group_name = f"group_{k}"
        indices = partition_data[group_name]
        
        # Get pairs
        pairs = gather_pairs(flat_params, indices)
        
        # Plot SDF contour
        sdf_fn = DIGIT_SDFS[k]
        # vmap over grid
        # We need to flatten grid, apply, reshape
        flat_grid = points_jax.reshape(-1, 2)
        # Use vmap if possible, but SDFs are vectorized on last dim usually
        # Let's try direct application
        # If flat_grid is (N, 2), SDF should return (N,)
        Z = sdf_fn(flat_grid)
        Z = np.array(Z).reshape(grid_res, grid_res)
        
        # Plot filled contour for shape
        # Negative values are inside
        ax.contourf(X, Y, Z, levels=[-10, 0], colors=['#e0f7fa'], alpha=0.5)
        ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
        
        # Scatter weights
        # Subsample if too many
        if len(pairs) > 1000:
            indices_sub = np.random.choice(len(pairs), 1000, replace=False)
            pairs_plot = pairs[indices_sub]
        else:
            pairs_plot = pairs
            
        ax.scatter(pairs_plot[:, 0], pairs_plot[:, 1], s=1, alpha=0.6, c='red', label='Weights')
        
        ax.set_title(f"Digit {k}\n({len(pairs)} pairs)")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('equal')
        # ax.axis('off')

    plt.tight_layout()
    plt.savefig("constrained_weights.png", dpi=150)
    print("Saved visualization to constrained_weights.png")

if __name__ == "__main__":
    main()
