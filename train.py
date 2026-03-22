import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import time

from data_loader import get_mnist_data
from geometry import DIGIT_SDFS
from alm import compute_alm_loss, update_multipliers

# --- Model Definition ---

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# --- Utilities ---

def get_params_flat(params):
    leaves, _ = jax.tree_util.tree_flatten(params)
    # Concatenate all leaves into a single 1D array
    return jnp.concatenate([x.ravel() for x in leaves])

def partition_indices(n_params, seed=0):
    # We need pairs, so n_params must be even ideally.
    # If odd, we ignore the last parameter.
    n_pairs = n_params // 2
    
    key = jax.random.PRNGKey(seed)
    shuffled_indices = jax.random.permutation(key, n_pairs)
    
    # Split into 10 groups
    # We use array_split to handle uneven divisions
    groups = jnp.array_split(shuffled_indices, 10)
    return groups

def gather_pairs(flat_params, group_indices):
    # flat_params: (N,)
    # group_indices: (M,) indices of pairs
    # We want to extract pairs (2*i, 2*i+1) for each i in group_indices
    
    # Construct indices for gather
    # For each pair index k, we want 2*k and 2*k+1
    idx_0 = 2 * group_indices
    idx_1 = 2 * group_indices + 1
    
    # Stack to get (M, 2)
    # But gather requires a single index array if we want 1D output.
    # Actually, jnp.take is easier.
    p0 = jnp.take(flat_params, idx_0)
    p1 = jnp.take(flat_params, idx_1)
    
    return jnp.stack([p0, p1], axis=-1)

# --- Training Logic ---

def create_train_state(rng, learning_rate):
    model = MLP()
    params = model.init(rng, jnp.ones([1, 784]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def compute_repulsion(pairs, subset_size=256):
    """
    Computes a log-potential based repulsion: V = sum( -log(dist + eps) )
    log potential has better long-range spreading properties.
    """
    n = pairs.shape[0]
    # Simple slice is fast
    subset = pairs[:subset_size]
    
    diff = subset[:, None, :] - subset[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    
    eps = 1e-6
    # Log-potential repulsion
    repulsion = -0.5 * jnp.log(dist_sq + eps)
    # mask diagonal
    repulsion = repulsion * (1.0 - jnp.eye(subset.shape[0]))
    return jnp.mean(repulsion)

def compute_grid_coverage(pairs, sdf_fn, grid_size=16):
    """
    Encourages points to cover the entire SDF region by penalizing 'empty' grid cells.
    Uses a smooth masking approach to stay JIT-friendly.
    """
    ticks = jnp.linspace(-1.0, 1.0, grid_size)
    X, Y = jnp.meshgrid(ticks, ticks)
    grid_points = jnp.stack([X, Y], axis=-1).reshape(-1, 2)
    
    # Smooth indicator: 1 if inside, 0 if outside
    # Using sigmoid of -sdf to get 1 for sdf < 0
    is_inside = jax.nn.sigmoid(-20.0 * sdf_fn(grid_points))
    
    subset_pairs = pairs[:512]
    diff = grid_points[:, None, :] - subset_pairs[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    
    # For each grid point, softmin of distances to weights
    # If no weight is close, min_dist_sq will be large.
    min_dist_sq = -0.1 * jnp.log(jnp.sum(jnp.exp(-10.0 * dist_sq), axis=-1) + 1e-6)
    
    # Only penalize 'inside' points
    weighted_penalty = min_dist_sq * is_inside
    return jnp.sum(weighted_penalty) / (jnp.sum(is_inside) + 1e-6)

@jax.jit
def train_step(state, batch, multipliers, mu, partition_groups):
    images, labels = batch
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        task_loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        
        flat = get_params_flat(params)
        
        total_penalty = 0.0
        total_repulsion = 0.0
        total_coverage = 0.0
        
        all_c_vals = []
        for k in range(10):
            pairs = gather_pairs(flat, partition_groups[k])
            c_val = DIGIT_SDFS[k](pairs)
            all_c_vals.append(c_val)
            
            # Repulsion (long-range spread)
            total_repulsion += compute_repulsion(pairs, subset_size=256)
            
            # Coverage (fill the whole shape)
            total_coverage += compute_grid_coverage(pairs, DIGIT_SDFS[k], grid_size=12)

        # ALM Penalty Term
        def penalty_element(c, lam):
            shifted = jnp.maximum(0.0, lam + mu * c)
            return (shifted**2 - lam**2) / (2 * mu)
        
        penalty_tree = jax.tree_util.tree_map(penalty_element, all_c_vals, multipliers)
        penalty = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y), penalty_tree, 0.0)
        
        # Combined Geometric Loss
        # Weights tuned for strong uniformity
        geom_loss = 1e-2 * total_repulsion + 0.5 * total_coverage
        
        return task_loss + penalty + geom_loss, (task_loss, penalty, total_repulsion, total_coverage)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (task_loss, penalty, repulsion, coverage)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, total_loss, task_loss, penalty, repulsion, coverage

@jax.jit
def update_multipliers_step(params, multipliers, mu, partition_groups):
    flat = get_params_flat(params)
    c_vals = []
    for k in range(10):
        pairs = gather_pairs(flat, partition_groups[k])
        dist = DIGIT_SDFS[k](pairs)
        c_vals.append(dist)
    
    new_multipliers = update_multipliers(multipliers, c_vals, mu)
    # Calculate max violation for logging
    max_violation = jnp.max(jnp.array([jnp.max(c) for c in c_vals]))
    return new_multipliers, max_violation

def main():
    # 1. Data
    print("Loading data...")
    train_images, train_labels, test_images, test_labels = get_mnist_data()
    train_images = jnp.array(train_images)
    train_labels = jnp.array(train_labels)
    
    # 2. Model & Partition
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    learning_rate = 1e-3
    state = create_train_state(init_rng, learning_rate)
    
    print("Partitioning weights...")
    flat_params = get_params_flat(state.params)
    n_params = flat_params.shape[0]
    print(f"Total parameters: {n_params}")
    
    partition_groups = partition_indices(n_params, seed=42)
    
    # Initialize multipliers (zeros)
    multipliers = []
    for group in partition_groups:
        multipliers.append(jnp.zeros(group.shape[0]))
        
    mu = 1.0
    
    # 3. Training Loop
    batch_size = 128
    epochs = 20
    steps_per_epoch = len(train_images) // batch_size
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Shuffle data
        rng, shuffle_rng = jax.random.split(rng)
        perms = jax.random.permutation(shuffle_rng, len(train_images))
        train_images = train_images[perms]
        train_labels = train_labels[perms]
        
        epoch_loss = 0
        epoch_task = 0
        epoch_pen = 0
        epoch_rep = 0
        epoch_cov = 0
        
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size
            batch = (train_images[start:end], train_labels[start:end])
            
            state, loss, task, pen, rep, cov = train_step(state, batch, multipliers, mu, partition_groups)
            
            epoch_loss += loss
            epoch_task += task
            epoch_pen += pen
            epoch_rep += rep
            epoch_cov += cov
        
        # Update multipliers & mu at end of epoch
        multipliers, max_viol = update_multipliers_step(state.params, multipliers, mu, partition_groups)
        
        # Increase mu
        mu *= 1.1
        
        avg_loss = epoch_loss / steps_per_epoch
        avg_task = epoch_task / steps_per_epoch
        avg_pen = epoch_pen / steps_per_epoch
        avg_rep = epoch_rep / steps_per_epoch
        avg_cov = epoch_cov / steps_per_epoch
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} (Task: {avg_task:.4f}, Pen: {avg_pen:.4f}, Rep: {avg_rep:.4f}, Cov: {avg_cov:.4f}) | "
              f"Max Viol: {max_viol:.4f} | Mu: {mu:.2f} | "
              f"Time: {time.time() - start_time:.2f}s")

    # Save final weights for visualization
    jnp.save("final_params.npy", get_params_flat(state.params))
    # Also save partition
    # partition_groups is a list of arrays, save as a dict or multiple files
    partition_dict = {f"group_{i}": np.array(g) for i, g in enumerate(partition_groups)}
    np.savez("partition.npz", **partition_dict)
    print("Training complete. Saved params and partition.")

if __name__ == "__main__":
    main()
