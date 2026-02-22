import gymnasium as gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from dev_PIDController import PIDController

# =================================================================
# 1. Simulation Execution
# =================================================================
# Create CartPole environment and POSITION ONLY controller
env = gym.make('CartPole-v1', render_mode='human')
dt = 1.0 / 50.0
pid_x = PIDController(kp=2.0, ki=0.0, kd=0.0, dt=dt)     # Cart position PID only

env.x_threshold = 4.0           # Wider cart bounds (±4.0m)
env.theta_threshold_radians = 0.418  # Wider angle (±24°)

state, _ = env.reset()

print("CartPole - POSITION ONLY PID Control!")
print("Target: Keep cart centered at x=0")
print("Close window to exit")

episode_score = 0
for t in range(10000):
    x, x_dot, theta, theta_dot = state
    
    # POSITION ONLY control - keep cart centered
    x_error = 0.0 - x
    action = pid_x.update(x_error)
    
    # Convert to discrete action (0=left, 1=right)
    action_discrete = int(np.clip(action > 0, 0, 1))
    
    state, reward, terminated, truncated, _ = env.step(action_discrete)
    episode_score += 1
    
    # Reset if pole falls
    if abs(theta) > 0.909:
        print(f"Episode ended at {episode_score} steps (θ={np.degrees(theta):.1f}°)")
        # Custom reset bounds (default: low=-0.05, high=0.05)
        state, info = env.reset()
        episode_score = 0
        pid_x.integral = 0.0  # Reset integral
    
    env.render()

env.close()


# =================================================================
# 2. Post-Processing & Analysis
# =================================================================
states = np.array(states)
actions = np.array(actions)

# Extract components for visualization
angles = np.arctan2(states[:, 1], states[:, 0])
velocities = states[:, 2]

# =================================================================
# 3. Visualization
# =================================================================
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot Angle: Shows how close the pendulum gets to 'Upright' (0 rad)
axs[0].plot(angles, label='Angle (rad)', color='royalblue', linewidth=1.5)
axs[0].axhline(y=0, color='r', linestyle='--', alpha=0.6, label='Target (Upright)')
axs[0].set_ylabel('Radians')
axs[0].set_title('Pendulum State: Angle')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# Plot Angular Velocity: Shows the stability/oscillation of the pole
axs[1].plot(velocities, label='Angular Velocity', color='darkorange', linewidth=1.5)
axs[1].set_ylabel('Rad/s')
axs[1].set_title('Pendulum State: Velocity')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

# Plot Control Actions: Shows the torque output (The PID effort)
axs[2].plot(actions, label='PID Torque', color='forestgreen', linewidth=1.5)
axs[2].set_ylabel('Torque (Nm)')
axs[2].set_xlabel('Time Steps')
axs[2].set_title('Controller Output (Control Effort)')
axs[2].grid(True, alpha=0.3)
axs[2].legend()

plt.tight_layout()
plt.show()