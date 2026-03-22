import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# --- Geometry Primitives & Operations ---

def smooth_min(a, b, k=32):
    return -jnp.logaddexp(-k * a, -k * b) / k

def smooth_max(a, b, k=32):
    return jnp.logaddexp(k * a, k * b) / k

def union(d1, d2, k=32):
    return smooth_min(d1, d2, k)

def intersection(d1, d2, k=32):
    return smooth_max(d1, d2, k)

def difference(d1, d2, k=32):
    return intersection(d1, -d2, k)

def safe_norm(x, axis=-1, eps=1e-8):
    return jnp.sqrt(jnp.sum(x**2, axis=axis) + eps)

def sd_circle(p, center, radius):
    return safe_norm(p - center, axis=-1) - radius

def sd_segment(p, a, b, thickness):
    pa = p - a
    ba = b - a
    h = jnp.clip(jnp.dot(pa, ba) / jnp.dot(ba, ba), 0.0, 1.0)
    return safe_norm(pa - ba * h[..., None], axis=-1) - thickness

def sd_arc(p, center, radius, angle_start, angle_end, thickness):
    # Simplified arc: distance to circle, but clipped by angle?
    # Easier: Union of many small segments approximating the arc
    # OR: just use segments for now for simplicity, or compositions of circles/boxes.
    # For robust smooth SDFs, composing segments is safest and easiest to debug.
    pass

# --- Digit Definitions ---

# 0: Hollow Ellipse (already defined in test)
def sdf_0(p):
    outer = sd_circle(p, jnp.array([0.0, 0.0]), 0.8)
    inner = sd_circle(p, jnp.array([0.0, 0.0]), 0.4)
    return difference(outer, inner)

# 1: Vertical Line
def sdf_1(p):
    return sd_segment(p, jnp.array([0.0, -0.8]), jnp.array([0.0, 0.8]), 0.15)

# 2: Top curve, diagonal, bottom line
def sdf_2(p):
    top_curve = sd_segment(p, jnp.array([-0.5, 0.5]), jnp.array([0.5, 0.5]), 0.15) # Simplification
    # Let's make it look better with segments
    s1 = sd_segment(p, jnp.array([-0.5, 0.6]), jnp.array([0.5, 0.6]), 0.15) # Top
    s2 = sd_segment(p, jnp.array([0.5, 0.6]), jnp.array([0.5, 0.3]), 0.15)  # Side down
    s3 = sd_segment(p, jnp.array([0.5, 0.3]), jnp.array([-0.5, -0.6]), 0.15) # Diagonal
    s4 = sd_segment(p, jnp.array([-0.5, -0.6]), jnp.array([0.5, -0.6]), 0.15) # Bottom
    return union(union(union(s1, s2), s3), s4)

# 3: Top bar, mid bar, bot bar, right side connections
def sdf_3(p):
    top = sd_segment(p, jnp.array([-0.5, 0.7]), jnp.array([0.5, 0.7]), 0.15)
    mid = sd_segment(p, jnp.array([-0.3, 0.0]), jnp.array([0.5, 0.0]), 0.15)
    bot = sd_segment(p, jnp.array([-0.5, -0.7]), jnp.array([0.5, -0.7]), 0.15)
    side_top = sd_segment(p, jnp.array([0.5, 0.7]), jnp.array([0.5, 0.0]), 0.15)
    side_bot = sd_segment(p, jnp.array([0.5, 0.0]), jnp.array([0.5, -0.7]), 0.15)
    return union(union(union(union(top, mid), bot), side_top), side_bot)

# 4: Vertical, diagonal, horizontal
def sdf_4(p):
    v = sd_segment(p, jnp.array([0.3, -0.8]), jnp.array([0.3, 0.8]), 0.15)
    diag = sd_segment(p, jnp.array([0.3, 0.8]), jnp.array([-0.5, 0.0]), 0.15)
    h = sd_segment(p, jnp.array([-0.5, 0.0]), jnp.array([0.5, 0.0]), 0.15)
    return union(union(v, diag), h)

# 5: Top, vertical, mid, right down, bot
def sdf_5(p):
    top = sd_segment(p, jnp.array([-0.5, 0.7]), jnp.array([0.5, 0.7]), 0.15)
    v_top = sd_segment(p, jnp.array([-0.5, 0.7]), jnp.array([-0.5, 0.1]), 0.15)
    mid = sd_segment(p, jnp.array([-0.5, 0.1]), jnp.array([0.4, 0.1]), 0.15)
    v_bot = sd_segment(p, jnp.array([0.4, 0.1]), jnp.array([0.4, -0.6]), 0.15)
    bot = sd_segment(p, jnp.array([0.4, -0.6]), jnp.array([-0.5, -0.6]), 0.15)
    return union(union(union(union(top, v_top), mid), v_bot), bot)

# 6: Circle bottom, line up
def sdf_6(p):
    # Hollow circle at bottom
    bot_outer = sd_circle(p, jnp.array([0.0, -0.4]), 0.45)
    bot_inner = sd_circle(p, jnp.array([0.0, -0.4]), 0.15)
    loop = difference(bot_outer, bot_inner)
    
    # Line up and curve top
    line = sd_segment(p, jnp.array([-0.35, -0.2]), jnp.array([-0.3, 0.6]), 0.15)
    top = sd_segment(p, jnp.array([-0.3, 0.6]), jnp.array([0.4, 0.7]), 0.15)
    return union(union(loop, line), top)

# 7: Top bar, diagonal down
def sdf_7(p):
    top = sd_segment(p, jnp.array([-0.6, 0.7]), jnp.array([0.6, 0.7]), 0.15)
    diag = sd_segment(p, jnp.array([0.6, 0.7]), jnp.array([-0.2, -0.8]), 0.15)
    return union(top, diag)

# 8: Two loops (defined in test)
def sdf_8(p):
    top_outer = sd_circle(p, jnp.array([0.0, 0.45]), 0.35)
    top_inner = sd_circle(p, jnp.array([0.0, 0.45]), 0.15)
    bot_outer = sd_circle(p, jnp.array([0.0, -0.45]), 0.40)
    bot_inner = sd_circle(p, jnp.array([0.0, -0.45]), 0.15)
    
    top = difference(top_outer, top_inner)
    bot = difference(bot_outer, bot_inner)
    return union(top, bot)

# 9: Loop top, line down
def sdf_9(p):
    # Hollow circle at top
    top_outer = sd_circle(p, jnp.array([0.0, 0.4]), 0.45)
    top_inner = sd_circle(p, jnp.array([0.0, 0.4]), 0.15)
    loop = difference(top_outer, top_inner)
    
    # Line down
    line = sd_segment(p, jnp.array([0.35, 0.2]), jnp.array([0.3, -0.6]), 0.15)
    bot = sd_segment(p, jnp.array([0.3, -0.6]), jnp.array([-0.4, -0.7]), 0.15)
    return union(union(loop, line), bot)

# Collection
DIGIT_SDFS = [sdf_0, sdf_1, sdf_2, sdf_3, sdf_4, sdf_5, sdf_6, sdf_7, sdf_8, sdf_9]

def plot_digits(save_path="digits_check.png"):
    # Grid for plotting
    x = jnp.linspace(-1.5, 1.5, 100)
    y = jnp.linspace(-1.5, 1.5, 100)
    X, Y = jnp.meshgrid(x, y)
    points = jnp.stack([X, Y], axis=-1)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, sdf in enumerate(DIGIT_SDFS):
        # Apply SDF
        # We need to map over the grid. Since our SDFs handle vectorization on the last dim (points),
        # we can just pass points directly if written correctly.
        # But `sd_segment` uses dot products which might need care with broadcasting.
        # Let's use vmap to be safe for the grid evaluation if needed, or rely on broadcasting.
        # However, our primitives use `axis=-1` norms, which should handle (H, W, 2).
        # `dot` in sd_segment: dot(pa, ba). pa is (H, W, 2), ba is (2,). 
        # Standard jnp.dot might behave differently. Let's use a safe wrapper or vmap.
        
        # Let's define a safe vectorized wrapper
        v_sdf = jax.vmap(jax.vmap(sdf))
        Z = v_sdf(points)
        
        ax = axes[i]
        c = ax.contourf(X, Y, Z, levels=20, cmap='RdBu')
        ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
        ax.set_title(f"Digit {i}")
        ax.axis('equal')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved digit visualization to {save_path}")

if __name__ == "__main__":
    plot_digits()
