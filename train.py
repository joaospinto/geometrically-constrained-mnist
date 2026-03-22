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

@jax.jit
def train_step(state, batch, multipliers, mu, partition_groups):
    """
    state: TrainState
    batch: (images, labels)
    multipliers: list of 10 arrays
    mu: scalar
    partition_groups: list of 10 arrays of pair indices
    """
    images, labels = batch
    
    def loss_fn(params):
        # Task Loss
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        task_loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        
        # ALM Penalty
        def constraints_wrapper(p):
            flat = get_params_flat(p)
            c_list = []
            for k in range(10):
                # Get pairs for digit k
                pairs = gather_pairs(flat, partition_groups[k])
                # Apply SDF
                dist = DIGIT_SDFS[k](pairs)
                c_list.append(dist)
            return c_list
            
        penalty, _ = compute_alm_loss(params, multipliers, mu, constraints_wrapper)
        
        return task_loss + penalty, (task_loss, penalty)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (task_loss, penalty)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, total_loss, task_loss, penalty

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
    epochs = 10
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
        
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size
            batch = (train_images[start:end], train_labels[start:end])
            
            state, loss, task, pen = train_step(state, batch, multipliers, mu, partition_groups)
            
            epoch_loss += loss
            epoch_task += task
            epoch_pen += pen
        
        # Update multipliers & mu at end of epoch
        multipliers, max_viol = update_multipliers_step(state.params, multipliers, mu, partition_groups)
        
        # Increase mu
        mu *= 1.1
        
        avg_loss = epoch_loss / steps_per_epoch
        avg_task = epoch_task / steps_per_epoch
        avg_pen = epoch_pen / steps_per_epoch
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f} (Task: {avg_task:.4f}, Pen: {avg_pen:.4f}) | "
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
