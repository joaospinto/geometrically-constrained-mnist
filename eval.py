import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from data_loader import get_mnist_data

# --- Model Definition (Must match train.py) ---

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

def reconstruct_params(flat_params):
    # We initialize a dummy model to get the structure (PyTree)
    model = MLP()
    variables = model.init(jax.random.PRNGKey(0), jnp.ones([1, 784]))
    params = variables['params']
    
    # Flatten the template to get the structure and sizes
    flat_template, treedef = jax.tree_util.tree_flatten(params)
    
    # Slice the flat_params array and reshape to match each leaf
    new_leaves = []
    pointer = 0
    for leaf in flat_template:
        size = leaf.size
        new_leaf = flat_params[pointer:pointer + size].reshape(leaf.shape)
        new_leaves.append(new_leaf)
        pointer += size
        
    return jax.tree_util.tree_unflatten(treedef, new_leaves)

def main():
    print("Loading data...")
    _, _, test_images, test_labels = get_mnist_data()
    test_images = jnp.array(test_images)
    test_labels = jnp.array(test_labels)

    print("Loading weights...")
    try:
        flat_params = jnp.load("final_params.npy")
    except FileNotFoundError:
        print("Error: final_params.npy not found. Please run train.py first.")
        return

    params = reconstruct_params(flat_params)
    model = MLP()

    # Evaluation
    print("Evaluating...")
    logits = model.apply({'params': params}, test_images)
    
    # Accuracy
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == test_labels)
    
    # Loss
    one_hot = jax.nn.one_hot(test_labels, 10)
    # Using optax-like cross entropy manually for simplicity in eval
    def softmax_cross_entropy(logits, labels):
        return -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    
    loss = jnp.mean(softmax_cross_entropy(logits, one_hot))

    print("-" * 30)
    print(f"Test Accuracy: {accuracy:.2%} ({jnp.sum(predictions == test_labels)}/{len(test_labels)})")
    print(f"Test Loss:     {loss:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
