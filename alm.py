import jax
import jax.numpy as jnp

def compute_alm_loss(params, multipliers, mu, constraints_fn):
    """
    Computes the Augmented Lagrangian term for inequality constraints c(x) <= 0.
    
    Args:
        params: Model parameters (PyTree).
        multipliers: Lagrange multipliers (PyTree matching constraints_fn output).
        mu: Penalty parameter (scalar).
        constraints_fn: Function taking params and returning a PyTree of constraint violations.
                        Positive values indicate violation.
    
    Returns:
        total_penalty: Scalar penalty term.
        c_vals: The computed constraint values (PyTree).
    """
    # Evaluate constraints
    c_vals = constraints_fn(params)
    
    def penalty_element(c, lam):
        # max(0, lam + mu * c)^2
        shifted = jnp.maximum(0.0, lam + mu * c)
        return (shifted**2 - lam**2) / (2 * mu)
    
    # Sum over all constraints
    penalty_tree = jax.tree_util.tree_map(penalty_element, c_vals, multipliers)
    
    # Sum all leaves
    total_penalty = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y), penalty_tree, 0.0)
    
    return total_penalty, c_vals

def update_multipliers(multipliers, c_vals, mu):
    """
    Update rule: lambda <- max(0, lambda + mu * c(x))
    """
    def update_element(lam, c):
        return jnp.maximum(0.0, lam + mu * c)
    
    return jax.tree_util.tree_map(update_element, multipliers, c_vals)
