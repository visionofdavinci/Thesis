"""
Drone Navigation Using Potential Fields + Residual PPO

This is the deployment-time integration with pybullet-drones. It connects
the potential field engines and PPO escape policy to the CtrlAviary
environment through a PID controller.

The navigation pipeline at each timestep:
1. Get current drone position from the physics simulation
2. Compute the potential field gradient at that position
3. Scale the gradient into a base velocity, modulated by dPhi/dt
4. Query the PPO for a residual correction, gated by threat level
5. Blend base velocity and PPO correction
6. Feed the resulting target position to the PID controller
7. Apply the PID-computed RPMs to the drone motors
"""

import numpy as np
import matplotlib.pyplot as plt
import time

#pybullet-drones components
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

#field engines and PPO components
from engines.subharmonic_field_engine import SubharmonicFieldEngine
from engines.superharmonic_field_engine import SuperharmonicFieldEngine
from engines.ppo_policy import PPOAgent
from engines.navigation_controller import build_escape_observation


class PotentialFieldNavigator:
    """
    Full deployment navigator: potential fields + PID + residual PPO.

    Uses the 3D SuperharmonicFieldEngine for consistent training/deployment.
    The PID controller converts target positions into motor RPMs.
    The residual PPO correction is gated by obstacle threat level.

    Parameters:
    - start_pos : np.ndarray - starting position [x, y, z]
    - goal_pos : np.ndarray - goal position [x, y, z]
    - workspace_bounds : tuple - ((x_min, x_max), (y_min, y_max))
    - target_altitude : float - nominal flight altitude
    - gui : bool - whether to show the pybullet GUI
    - record_video : bool - whether to record video
    - ppo_checkpoint : str - path to PPO checkpoint directory
    """

    def __init__(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                 workspace_bounds: tuple = ((0, 10), (0, 10)),
                 target_altitude: float = 1.0, gui: bool = True,
                 record_video: bool = False, ppo_checkpoint: str = ''):

        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.target_altitude = target_altitude
        self.workspace_bounds = workspace_bounds

        #initialize CtrlAviary environment
        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=start_pos.reshape(1, 3),
            initial_rpys=np.array([[0, 0, 0]]),
            physics=Physics.PYB,
            neighbourhood_radius=10,
            pyb_freq=240,
            ctrl_freq=48,
            gui=gui,
            record=record_video,
            obstacles=False,
            user_debug_gui=False
        )

        #PID controller for converting target positions to RPMs
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

        #3D analytical field engine (matches PPO training config)
        self.field_engine = SuperharmonicFieldEngine(
            goal_position=goal_pos,
            a_att=0.1,
            a_rep=1.0,
            n_power=2.0,
            danger_distance=1.0,
        )

        #control parameters
        self.velocity_gain = 2.0
        self.max_velocity = 1.5
        self.goal_tolerance = 0.3

        #navigation parameters
        self.min_gradient_threshold = 0.01  #flat region detection threshold
        self.stuck_counter = 0
        self.last_distance_to_goal = None

        #dPhi/dt speed modulator parameters
        self.dphi_dt_gain = 0.4      #beta: how strongly dPhi/dt modulates speed
        self.min_speed_scale = 0.2   #never drop below 20% of nominal speed
        self.stuck_threshold = 40    #steps before PPO rescue

        #residual PPO parameters
        self.ppo_escape_speed = 1.2
        self.initial_d_goal = np.linalg.norm(start_pos - goal_pos)

        #load PPO agent (obs_dim = 12, field-only observations)
        obs_dim = 12
        self.ppo_agent = PPOAgent(obs_dim=obs_dim, act_dim=3, hidden_sizes=[64, 64])
        if ppo_checkpoint:
            import os
            if os.path.exists(ppo_checkpoint):
                self.ppo_agent.load(ppo_checkpoint)
                print(f"  Loaded PPO checkpoint from {ppo_checkpoint}")
            else:
                print(f"  Warning: PPO checkpoint {ppo_checkpoint} not found, "
                      f"using random policy")

        #trajectory tracking
        self.trajectory = []
        self.time_steps = []
        self.current_time = 0.0
        self.dt = 1.0 / self.env.CTRL_FREQ
        self.step_count = 0

        #target position for PID
        self.target_position = start_pos.copy()

        #simulation state
        self.reached_goal = False
        self.max_simulation_time = 800.0

        #debug output control
        self.debug = True
        self.debug_counter = 0

    def add_obstacle(self, position: np.ndarray, radius: float,
                     velocity: np.ndarray = None):
        """
        Adds a 3D obstacle to the field engine.

        Parameters:
        - position : np.ndarray - [x, y, z] position of obstacle centre
        - radius : float - collision radius
        - velocity : np.ndarray, optional - [vx, vy, vz] velocity
        Returns: none
        """
        self.field_engine.add_obstacle(
            position=position,
            radius=radius,
            velocity=velocity
        )

    def compute_target_position(self, current_pos: np.ndarray) -> np.ndarray:
        """
        Computes next target position using 3D gradient + residual PPO
        gated by threat level.

        The pipeline:
        1. Compute the negative gradient as the base velocity direction
        2. Scale by velocity_gain, cap at max_velocity
        3. Modulate speed by dPhi/dt (slow down when obstacle approaching)
        4. Compute threat level from minimum obstacle distance
        5. If threat > 0.01, query PPO for a residual correction
        6. Blend: (1-threat)*base_velocity + threat*ppo_correction

        Parameters:
        - current_pos : np.ndarray - current drone [x, y, z] position
        Returns:
        - new_target : np.ndarray - next target [x, y, z] for the PID
        """

        #3D analytical gradient descent (base velocity)
        gradient = self.field_engine.compute_gradient(current_pos)
        current_potential = self.field_engine.compute_potential(current_pos)
        gradient_magnitude = np.linalg.norm(gradient)

        #check if drone is making progress (for stuck detection)
        current_distance_to_goal = np.linalg.norm(current_pos - self.goal_pos)
        if self.last_distance_to_goal is not None:
            progress = self.last_distance_to_goal - current_distance_to_goal
            if progress < 0.01 * self.dt:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 2)
        self.last_distance_to_goal = current_distance_to_goal

        #compute base velocity from negative gradient
        base_velocity = -gradient
        vel_mag = np.linalg.norm(base_velocity)
        if vel_mag > 1e-6:
            base_velocity = (base_velocity / vel_mag
                             * min(vel_mag * self.velocity_gain, self.max_velocity))
        else:
            base_velocity = np.zeros(3)

        #dPhi/dt speed modulator: slow down when an obstacle is approaching
        dphi_dt = self.field_engine.compute_dphi_dt(current_pos)
        speed_scale = 1.0 - self.dphi_dt_gain * max(dphi_dt, 0.0)
        speed_scale = max(speed_scale, self.min_speed_scale)
        base_velocity *= speed_scale

        #residual PPO correction, gated by threat level
        obstacles = self.field_engine.obstacles
        if len(obstacles) > 0:
            min_obs_dist = min(
                np.linalg.norm(current_pos - o.position) - o.radius
                for o in obstacles
            )
            min_obs_dist = max(min_obs_dist, 0.0)
        else:
            min_obs_dist = float('inf')

        danger_distance = self.field_engine.danger_distance
        threat_level = np.clip(1.0 - min_obs_dist / danger_distance, 0.0, 1.0)

        #override to full PPO when stuck
        if self.stuck_counter > self.stuck_threshold:
            threat_level = 1.0

        #build observation and query PPO (skip when no threat to save compute)
        if threat_level > 0.01:
            obs = build_escape_observation(
                agent_pos=current_pos,
                field_engine=self.field_engine,
                step_count=self.step_count,
                max_episode_steps=int(self.max_simulation_time / self.dt),
                initial_d_goal=self.initial_d_goal,
            )
            ppo_action, _, _, _ = self.ppo_agent.select_action(obs)
            ppo_correction = ppo_action * self.ppo_escape_speed

            #blend: base velocity when safe, PPO correction when threatened
            final_velocity = ((1.0 - threat_level) * base_velocity
                              + threat_level * ppo_correction)
        else:
            final_velocity = base_velocity

        #move to next position
        new_target = current_pos + final_velocity * self.dt
        new_target[2] = np.clip(new_target[2], 0.1, self.target_altitude + 1.0)

        #clamp to workspace bounds
        new_target[0] = np.clip(new_target[0],
                                self.workspace_bounds[0][0],
                                self.workspace_bounds[0][1])
        new_target[1] = np.clip(new_target[1],
                                self.workspace_bounds[1][0],
                                self.workspace_bounds[1][1])

        #debug output (every 48 steps = ~1 second at 48Hz)
        if self.debug and self.debug_counter % 48 == 0:
            print(f"\n[DEBUG] Current: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, "
                  f"{current_pos[2]:.3f}]")
            print(f"[DEBUG] Gradient: [{gradient[0]:.4f}, {gradient[1]:.4f}, "
                  f"{gradient[2]:.4f}] (mag: {gradient_magnitude:.4f})")
            print(f"[DEBUG] Potential: {current_potential:.4f} | "
                  f"Goal dist: {current_distance_to_goal:.3f}m")
            print(f"[DEBUG] dPhi/dt: {dphi_dt:+.4f} | speed_scale: {speed_scale:.2f} | "
                  f"stuck: {self.stuck_counter}")
            print(f"[DEBUG] Threat: {threat_level:.2f} | min_obs_dist: {min_obs_dist:.3f}m")
            print(f"[DEBUG] Target: [{new_target[0]:.3f}, {new_target[1]:.3f}, "
                  f"{new_target[2]:.3f}]")

        self.debug_counter += 1
        self.step_count += 1

        return new_target

    def check_goal_reached(self, position: np.ndarray) -> bool:
        """
        Checks if the drone has reached the goal.

        Parameters:
        - position : np.ndarray - current drone position
        Returns:
        - reached : bool
        """
        distance_to_goal = np.linalg.norm(position - self.goal_pos)
        return distance_to_goal < self.goal_tolerance

    def run_navigation(self, visualize_field: bool = True):
        """
        Runs 3D analytical potential field navigation with residual PPO.

        This is the main simulation loop. At each control step:
        1. Update obstacle positions
        2. Get drone state from physics simulation
        3. Check goal reached
        4. Compute target position from potential field + PPO
        5. Use PID to compute motor RPMs from target
        6. Apply RPMs and advance simulation

        Parameters:
        - visualize_field : bool - whether to show plots after navigation
        Returns:
        - success : bool - whether the goal was reached
        - trajectory : np.ndarray - recorded trajectory points
        """
        print("Starting 3D analytical navigation with residual PPO")
        print(f"Start: {self.start_pos}")
        print(f"Goal: {self.goal_pos}")
        print(f"Obstacles: {len(self.field_engine.obstacles)}")

        #test gradient at start
        test_grad = self.field_engine.compute_gradient(self.start_pos)
        test_pot = self.field_engine.compute_potential(self.start_pos)
        print(f"\n[INITIAL] Gradient at start: {test_grad}")
        print(f"[INITIAL] Potential at start: {test_pot:.4f}")

        #reset environment
        obs, info = self.env.reset()

        #initialize target position at start
        self.target_position = self.start_pos.copy()

        #main control loop
        step_count = 0
        start_time = time.time()

        while self.current_time < self.max_simulation_time:
            #advance obstacle positions (analytical, no grid solve)
            #this is the ONLY place obstacle positions are updated in
            #the deployment loop (the training env does it in env.step())
            for obs in self.field_engine.obstacles:
                obs.update_position(self.dt)

            #get current state vector for the drone
            state = self.env._getDroneStateVector(0)
            current_pos = state[:3]

            #record trajectory
            self.trajectory.append(current_pos.copy())
            self.time_steps.append(self.current_time)

            #check if goal reached
            if self.check_goal_reached(current_pos):
                self.reached_goal = True
                print(f"\nGoal reached in {self.current_time:.2f} seconds!")
                print(f"  Final position: {current_pos}")
                print(f"  Distance to goal: "
                      f"{np.linalg.norm(current_pos - self.goal_pos):.3f}m")
                break

            #compute next target position from potential field + PPO
            self.target_position = self.compute_target_position(current_pos)

            #use PID controller to compute motor RPMs from target position
            target_rpy = np.array([0, 0, 0])  #keep level
            rpm, _, _ = self.ctrl.computeControlFromState(
                control_timestep=self.dt,
                state=state,
                target_pos=self.target_position,
                target_rpy=target_rpy
            )

            #apply RPM action (CtrlAviary expects 4 RPM values)
            obs, reward, terminated, truncated, info = self.env.step(rpm.reshape(1, 4))

            #update time
            self.current_time += self.dt
            step_count += 1

            #print progress (every ~2 seconds)
            if step_count % (2 * self.env.CTRL_FREQ) == 0:
                dist_to_goal = np.linalg.norm(current_pos - self.goal_pos)
                potential = self.field_engine.compute_potential(current_pos)
                dist_to_target = np.linalg.norm(current_pos - self.target_position)
                print(f"t={self.current_time:.1f}s | "
                      f"pos=[{current_pos[0]:.2f}, {current_pos[1]:.2f}, "
                      f"{current_pos[2]:.2f}] | "
                      f"goal_dist={dist_to_goal:.2f}m | phi={potential:.3f} | "
                      f"tracking_err={dist_to_target:.3f}m")

            #check termination
            if terminated or truncated:
                print(f"Environment terminated at t={self.current_time:.2f}s")
                break

        #simulation ended
        if not self.reached_goal:
            print(f"\nGoal not reached within {self.max_simulation_time}s")
            if len(self.trajectory) > 0:
                final_dist = np.linalg.norm(self.trajectory[-1] - self.goal_pos)
                print(f"  Final distance to goal: {final_dist:.2f}m")

                distances = [np.linalg.norm(pos - self.goal_pos)
                             for pos in self.trajectory]
                if len(distances) > 10:
                    initial_dist = distances[0]
                    final_dist = distances[-1]
                    progress = (initial_dist - final_dist) / initial_dist * 100
                    print(f"  Progress made: {progress:.1f}%")

        elapsed_time = time.time() - start_time
        print(f"\nSimulation completed in {elapsed_time:.2f}s (real time)")
        print(f"Total trajectory points: {len(self.trajectory)}")

        #close environment
        self.env.close()

        #visualize
        if visualize_field and len(self.trajectory) > 0:
            self.visualize_results()

        return (self.reached_goal,
                np.array(self.trajectory) if len(self.trajectory) > 0
                else np.array([]))

    def visualize_results(self):
        """
        Plots a 2D potential field slice with trajectory overlay, a convergence
        plot, and a 3D trajectory plot.

        Parameters: none
        Returns: none
        """
        if len(self.trajectory) == 0:
            print("No trajectory to visualize!")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        #panel 1: 2D slice of potential field at target altitude + trajectory
        ax = axes[0]
        res = 80
        x_range = np.linspace(self.workspace_bounds[0][0],
                               self.workspace_bounds[0][1], res)
        y_range = np.linspace(self.workspace_bounds[1][0],
                               self.workspace_bounds[1][1], res)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        for i in range(res):
            for j in range(res):
                pos3d = np.array([X[i, j], Y[i, j], self.target_altitude])
                Z[i, j] = self.field_engine.compute_potential(pos3d)

        contour = ax.contourf(X, Y, Z, levels=50, cmap='coolwarm')
        fig.colorbar(contour, ax=ax, label='Potential Phi')

        #obstacles
        for obs_obj in self.field_engine.obstacles:
            circle = plt.Circle(obs_obj.position[:2], obs_obj.radius,
                                color='red', alpha=0.5)
            ax.add_patch(circle)

        #trajectory overlay
        trajectory_2d = np.array([pos[:2] for pos in self.trajectory])
        ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1],
                'b-', linewidth=2, label='Trajectory', alpha=0.8)
        ax.plot(self.start_pos[0], self.start_pos[1],
                'go', markersize=15, label='Start',
                markeredgecolor='white', markeredgewidth=2)
        ax.plot(self.goal_pos[0], self.goal_pos[1],
                'r*', markersize=18, label='Goal',
                markeredgecolor='white', markeredgewidth=2)
        ax.plot(trajectory_2d[-1, 0], trajectory_2d[-1, 1],
                'bs', markersize=12, label='End',
                markeredgecolor='white', markeredgewidth=2)

        ax.legend(loc='upper right')
        ax.set_title(f'Navigation {"SUCCESS!" if self.reached_goal else "Incomplete"}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')

        #panel 2: distance to goal over time
        ax2 = axes[1]
        distances = [np.linalg.norm(pos - self.goal_pos) for pos in self.trajectory]
        ax2.plot(self.time_steps, distances, 'b-', linewidth=1.5)
        ax2.axhline(self.goal_tolerance, color='g', linestyle='--',
                     label='Goal tolerance')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Distance to goal (m)')
        ax2.set_title('Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        #3D plot
        if len(self.trajectory) > 1:
            self.plot_3d_trajectory()

    def plot_3d_trajectory(self):
        """
        Creates a 3D trajectory plot with start, goal, and obstacles.
        Parameters: none
        Returns: none
        """
        trajectory_array = np.array(self.trajectory)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        #trajectory
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1],
                trajectory_array[:, 2],
                'b-', linewidth=2, label='Trajectory')

        #start and goal
        ax.scatter(self.start_pos[0], self.start_pos[1], self.start_pos[2],
                   c='green', s=200, marker='o', label='Start',
                   edgecolors='black', linewidths=2)
        ax.scatter(self.goal_pos[0], self.goal_pos[1], self.goal_pos[2],
                   c='red', s=200, marker='*', label='Goal',
                   edgecolors='black', linewidths=2)

        #obstacles (circles at several z-levels)
        for obstacle in self.field_engine.obstacles:
            theta = np.linspace(0, 2 * np.pi, 30)
            for z in [0, 1.5, 3]:
                x_circle = obstacle.position[0] + obstacle.radius * np.cos(theta)
                y_circle = obstacle.position[1] + obstacle.radius * np.sin(theta)
                z_circle = np.full_like(theta, z)
                ax.plot(x_circle, y_circle, z_circle, 'r-', alpha=0.3, linewidth=1)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# example scripts

def example_simple_navigation():
    """
    Example: Simple navigation to a goal with no obstacles.
    """
    print("\n" + "=" * 60)
    print("Potential Field Navigation with PID Control")
    print("=" * 60 + "\n")

    navigator = PotentialFieldNavigator(
        start_pos=np.array([1.0, 1.0, 0.5]),
        goal_pos=np.array([8.0, 8.0, 1.0]),
        workspace_bounds=((0, 10), (0, 10)),
        target_altitude=1.0,
        gui=True,
        record_video=False,
        ppo_checkpoint='ppo_escape_checkpoints'
    )

    success, trajectory = navigator.run_navigation(visualize_field=True)
    return navigator, success


def example_obstacle_avoidance():
    """
    Example: Obstacle avoidance with residual PPO.
    """
    print("\n" + "=" * 60)
    print("Obstacle Avoidance Navigation (3D + Residual PPO)")
    print("=" * 60 + "\n")

    navigator = PotentialFieldNavigator(
        start_pos=np.array([1.0, 1.0, 0.5]),
        goal_pos=np.array([9.0, 5.0, 1.0]),
        workspace_bounds=((0, 10), (0, 10)),
        target_altitude=1.0,
        gui=True,
        record_video=False,
        ppo_checkpoint='ppo_escape_checkpoints'
    )

    navigator.add_obstacle(position=np.array([3.5, 5.0, 1.0]), radius=0.6)
    navigator.add_obstacle(position=np.array([6.5, 5.0, 1.0]), radius=0.6)
    navigator.add_obstacle(position=np.array([9.0, 2.0, 1.0]), radius=0.6)

    success, trajectory = navigator.run_navigation(visualize_field=True)
    return navigator, success


if __name__ == "__main__":
    navigator, success = example_obstacle_avoidance()
    print("\nDone!")