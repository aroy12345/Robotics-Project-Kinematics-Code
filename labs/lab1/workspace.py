from lib.calculateFK import FK
from core.interfaces import ArmController
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fk = FK()
# the dictionary below contains the data returned by calling arm.joint_limits()
limits = [
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -1.7628, 'upper': 1.7628},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -3.0718, 'upper': -0.0698},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -0.0175, 'upper': 3.7525},
    {'lower': -2.8973, 'upper': 2.8973}
]

def generate_random_configuration():
    """Generate a random joint configuration within the limits."""
    return [np.random.uniform(limit['lower'], limit['upper']) for limit in limits]

def calculate_workspace_points(num_samples=10000):
    """Calculate end-effector positions for random joint configurations."""
    workspace_points = []
    for _ in range(num_samples):
        q = generate_random_configuration()
        joint_positions, _ = fk.forward(q)
        workspace_points.append(joint_positions[-1])  # Last point is the end-effector
    return np.array(workspace_points)

# Create plot to visualize the reachable workspace of the Panda arm
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Calculate workspace points
num_samples = 50000  # You can adjust this number for more or fewer points
workspace_points = calculate_workspace_points(num_samples)

# Plot workspace points
scatter = ax.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2], 
                     c=workspace_points[:, 2], cmap='viridis', marker='.', alpha=0.1)

# Plot robot base
ax.scatter(0, 0, 0, c='r', marker='o', s=100, label='Robot Base')

# Set labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'Panda Arm Reachable Workspace ({num_samples} samples)')

# Set equal aspect ratio
ax.set_box_aspect((1, 1, 1))

# Add a color bar
plt.colorbar(scatter, label='Z-axis position (m)')

plt.legend()
plt.tight_layout()
plt.show()