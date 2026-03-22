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

def sample_uniform_in_sdf(sdf_fn, n_pairs, rng):
    """
    Sample points uniformly within the region where sdf_fn(p) < 0.
    """
    points = []
    curr_rng = rng
    while len(points) < n_pairs:
        curr_rng, sub_rng = jax.random.split(curr_rng)
        # Sample in bounding box [-1.2, 1.2]
        candidates = jax.random.uniform(sub_rng, (n_pairs * 2, 2), minval=-1.2, maxval=1.2)
        dists = sdf_fn(candidates)
        inside = candidates[dists < 0]
        points.append(inside)
        n_collected = sum(p.shape[0] for p in points)
        if n_collected >= n_pairs:
            break
            
    all_points = jnp.concatenate(points, axis=0)
    return all_points[:n_pairs]

def create_train_state(rng, learning_rate, partition_groups):
    model = MLP()
    # Dummy init to get structure
    variables = model.init(rng, jnp.ones([1, 784]))
    params = variables['params']
    
    # Re-initialize weights uniformly within their SDFs
    flat_params = get_params_flat(params)
    new_flat = np.array(flat_params) # Use numpy for item assignment then back to jax
    
    print("Initializing weights uniformly within digit shapes...")
    init_rngs = jax.random.split(rng, 10)
    for k in range(10):
        indices = partition_groups[k]
        n_pairs = indices.shape[0]
        sampled_pairs = sample_uniform_in_sdf(DIGIT_SDFS[k], n_pairs, init_rngs[k])
        
        # Map back to flat_params
        idx_0 = 2 * indices
        idx_1 = 2 * indices + 1
        new_flat[idx_0] = sampled_pairs[:, 0]
        new_flat[idx_1] = sampled_pairs[:, 1]
    
    # Reconstruct PyTree
    leaves, treedef = jax.tree_util.tree_flatten(params)
    new_leaves = []
    pointer = 0
    for leaf in leaves:
        size = leaf.size
        new_leaves.append(jnp.array(new_flat[pointer:pointer+size]).reshape(leaf.shape))
        pointer += size
    new_params = jax.tree_util.tree_unflatten(treedef, new_leaves)
    
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=new_params, tx=tx)

def compute_repulsion(pairs, subset_size=256):
    """
    Stronger repulsion to prevent collapse.
    """
    subset = pairs[:subset_size]
    diff = subset[:, None, :] - subset[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    eps = 1e-5
    # Stronger 1/dist^2 repulsion for local spacing
    repulsion = 1.0 / (dist_sq + eps)
    repulsion = repulsion * (1.0 - jnp.eye(subset.shape[0]))
    return jnp.mean(repulsion)

def compute_grid_coverage(pairs, sdf_fn, grid_size=20):
    """
    High-resolution grid coverage to ensure full shape occupancy.
    """
    ticks = jnp.linspace(-1.1, 1.1, grid_size)
    X, Y = jnp.meshgrid(ticks, ticks)
    grid_points = jnp.stack([X, Y], axis=-1).reshape(-1, 2)
    is_inside = jax.nn.sigmoid(-30.0 * sdf_fn(grid_points))
    
    # Check distance of every grid point to its nearest weight
    # Subsampling pairs for speed but using more than before
    subset_pairs = pairs[:1024]
    diff = grid_points[:, None, :] - subset_pairs[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    
    # Penalty if nearest weight is far
    min_dist_sq = -0.05 * jnp.log(jnp.sum(jnp.exp(-20.0 * dist_sq), axis=-1) + 1e-6)
    return jnp.sum(min_dist_sq * is_inside) / (jnp.sum(is_inside) + 1e-6)

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
            
            total_repulsion += compute_repulsion(pairs, subset_size=256)
            total_coverage += compute_grid_coverage(pairs, DIGIT_SDFS[k], grid_size=16)

        def penalty_element(c, lam):
            shifted = jnp.maximum(0.0, lam + mu * c)
            return (shifted**2 - lam**2) / (2 * mu)
        
        penalty_tree = jax.tree_util.tree_map(penalty_element, all_c_vals, multipliers)
        penalty = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y), penalty_tree, 0.0)
        
        # Significantly higher weights for geometry to fight task-loss collapse
        geom_loss = 2e-3 * total_repulsion + 2.0 * total_coverage
        
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
    
    print("Partitioning weights...")
    # Dummy params to get size
    dummy_model = MLP()
    dummy_params = dummy_model.init(rng, jnp.ones([1, 784]))['params']
    flat_dummy = get_params_flat(dummy_params)
    n_params = flat_dummy.shape[0]
    print(f"Total parameters: {n_params}")
    
    partition_groups = partition_indices(n_params, seed=42)
    
    state = create_train_state(init_rng, learning_rate, partition_groups)
    
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
