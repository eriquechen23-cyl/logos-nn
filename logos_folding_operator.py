import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# ==========================================
# 1. Initial Universe Setup: Continuous XOR Potential Field
# ==========================================
np.random.seed(42)
N = 400
# Generate random points in the [-2, 2] range
X_original = np.random.uniform(-2, 2, (N, 2))

# Define Polarity Truth (XOR)
# Same-sign quadrants (+1): Deep Space Blue (Resonance)
# Opposite-sign quadrants (-1): Neon Red (Tension)
Y = np.sign(X_original[:, 0] * X_original[:, 1])
colors = np.where(Y > 0, '#00e5ff', '#ff0055')

# ==========================================
# 2. Target Coordinates for Absolute Potential Folding Operator f(x) = |x|
# ==========================================
X_target = np.zeros_like(X_original)
# Local Observer 1: Resonance Potential |x1 + x2|
X_target[:, 0] = np.abs(X_original[:, 0] + X_original[:, 1])
# Local Observer 2: Tension Potential |x1 - x2|
X_target[:, 1] = np.abs(X_original[:, 0] - X_original[:, 1])

# ==========================================
# 3. Build Deep Space Observatory (Canvas Setup)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0b0f19')
ax.set_facecolor('#0b0f19')

# Draw Galaxy Points
scatter = ax.scatter(X_original[:, 0], X_original[:, 1], 
                     c=colors, s=50, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=3)

# Draw Axes (Hide outer borders)
ax.axhline(0, color='#333333', linewidth=1, zorder=1)
ax.axvline(0, color='#333333', linewidth=1, zorder=1)
for spine in ax.spines.values():
    spine.set_visible(False)

# Set Title and Text
title = ax.set_title("Logos O(0) Space: Original XOR Topology", 
                     color='white', fontsize=16, pad=20, fontfamily='monospace')
ax.tick_params(colors='#555555')

# Decision Boundary (Singularity Cut Line), initially transparent
decision_line, = ax.plot([-1, 5], [-1, 5], color='#ffffff', linestyle='--', linewidth=2, alpha=0, zorder=2)

# ==========================================
# 4. Spatiotemporal Transition Function (Animation Update)
# ==========================================
frames = 120  # Total animation frames
pause_frames = 30 # Pause duration at the beginning and end

def update(frame):
    # Calculate Time Phase (0.0 to 1.0)
    if frame < pause_frames:
        alpha = 0.0
    elif frame > (frames - pause_frames):
        alpha = 1.0
    else:
        # True progress after deducting pause time
        progress = (frame - pause_frames) / (frames - 2 * pause_frames)
        # Use Smoothstep function for a more realistic physical transition
        alpha = progress**2 * (3 - 2 * progress)
        
    # Linear interpolation to calculate current coordinates
    current_X = (1 - alpha) * X_original + alpha * X_target
    scatter.set_offsets(current_X)
    
    # Dynamically adjust Field of View (FOV)
    current_xlim = -2.5 * (1 - alpha) + 0 * alpha
    current_xmax =  2.5 * (1 - alpha) + 5 * alpha
    current_ylim = -2.5 * (1 - alpha) + 0 * alpha
    current_ymax =  2.5 * (1 - alpha) + 5 * alpha
    
    ax.set_xlim(current_xlim, current_xmax)
    ax.set_ylim(current_ylim, current_ymax)
    
    # Dynamically update Text and Decision Boundary
    if alpha < 0.1:
        title.set_text("Logos O(0): Original XOR Field (Non-linearly Inseparable)")
        decision_line.set_alpha(0)
    elif alpha < 0.9:
        title.set_text(f"Applying Absolute Folding Operator f(x)=|x| ... ({alpha*100:.0f}%)")
        decision_line.set_alpha(0)
    else:
        title.set_text("Space Collapse Complete: Linearly Separable Singularity")
        # Gracefully fade in the decision boundary
        fade_in = (alpha - 0.9) * 10
        decision_line.set_alpha(fade_in)

    return scatter, title, decision_line

# ==========================================
# 5. Output the Truth (Save as GIF)
# ==========================================
print("[System] Rendering high-dimensional topological folding animation, please wait...")
ani = animation.FuncAnimation(fig, update, frames=frames, interval=40, blit=False)

# Save as GIF file (Requires pillow, usually built into matplotlib)
output_filename = 'logos_folding_operator.gif'
ani.save(output_filename, writer='pillow', fps=30)
print(f"[Observation Complete] Animation successfully exported to: {output_filename}")

# If using Jupyter Notebook, you can preview it with plt.show()
# plt.show()