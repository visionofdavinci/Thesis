"""
Drone Navigation Using Potential Fields - FIXED VERSION
Uses PID controller to compute RPMs from target positions
"""

import numpy as np
import matplotlib.pyplot as plt
import time

#import pybullet-drones components
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

#import your potential field engine
from potential_field_engine import PotentialFieldEngine2D, Obstacle


class PotentialFieldNavigator:
    
    def __init__(self, 
                 start_pos: np.ndarray,
                 goal_pos: np.ndarray,
                 workspace_bounds: tuple = ((0, 10), (0, 10)),
                 target_altitude: float = 1.0,
                 grid_resolution: int = 100,
                 gui: bool = True,
                 record_video: bool = False):
        
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
        
        #create PID controller for converting target positions to RPMs
        self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        
        #initialize the potential field engine
        self.field_engine = PotentialFieldEngine2D(
            x_bounds=workspace_bounds[0],
            y_bounds=workspace_bounds[1],
            grid_resolution=grid_resolution,
            goal_position=goal_pos[:2],
            max_iterations=2000,
            convergence_threshold=1e-3
        )
        
        #control parameters
        self.velocity_gain = 2.0
        self.max_velocity = 1.5
        self.goal_tolerance = 0.3
        
        #navigation parameters
        self.min_gradient_threshold = 0.01  #threshold for flat region detection
        self.stuck_counter = 0  #counts timesteps with low progress
        self.last_distance_to_goal = None
        
        #trajectory tracking
        self.trajectory = []
        self.time_steps = []
        self.current_time = 0.0
        self.dt = 1.0 / self.env.CTRL_FREQ
        
        #target position
        self.target_position = start_pos.copy()
        
        #simulation state
        self.reached_goal = False
        self.max_simulation_time = 800.0
        
        #debug
        self.debug = True
        self.debug_counter = 0
        
    def add_cylindrical_obstacle(self, center_xy: np.ndarray, radius: float, velocity: np.ndarray = None):
        self.field_engine.add_obstacle(
            position=center_xy,
            radius=radius,
            velocity=velocity
        )
        
    def compute_target_position(self, current_pos: np.ndarray) -> np.ndarray:
        #get 2D gradient and current potential
        pos_2d = current_pos[:2]
        gradient = self.field_engine.compute_gradient(pos_2d)
        current_potential = self.field_engine.compute_potential(pos_2d)
        gradient_magnitude = np.linalg.norm(gradient)
        
        #check if drone is making progress
        current_distance_to_goal = np.linalg.norm(pos_2d - self.goal_pos[:2])
        if self.last_distance_to_goal is not None:
            progress = self.last_distance_to_goal - current_distance_to_goal
            if progress < 0.01 * self.dt:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(0, self.stuck_counter - 2)
        self.last_distance_to_goal = current_distance_to_goal
        
        #compute velocity direction from negative gradient (gradient descent)
        velocity_direction = -gradient
        
        #normalize and scale velocity
        velocity_magnitude = np.linalg.norm(velocity_direction)
        if velocity_magnitude > 1e-6:
            velocity_direction = velocity_direction / velocity_magnitude
            velocity_magnitude = min(velocity_magnitude * self.velocity_gain, self.max_velocity)
        else:
            velocity_magnitude = 0.0
        
        #compute raw next move point from APF
        displacement_xy = velocity_direction * velocity_magnitude * self.dt
        raw_next_2d = pos_2d + displacement_xy
        
        #apply circular sampling technique (Section IV of paper) when:
        #  - in subharmonic mode
        #  - gradient is weak (flat region) or drone is stuck
        use_circular_sampling = (
            self.field_engine.subharmonic_mode and 
            (gradient_magnitude < self.min_gradient_threshold or self.stuck_counter > 50)
        )
        
        if use_circular_sampling:
            #circular sampling: sample points on circle around raw next point,
            #filter by danger distance, pick best by look-ahead evaluation
            adjusted_2d = self.field_engine.circular_sample(
                raw_next_point=raw_next_2d,
                current_pos=pos_2d,
                sample_radius=0.5,
                n_samples=16
            )
        else:
            adjusted_2d = raw_next_2d
        
        #build 3D target position
        new_target = current_pos.copy()
        new_target[0] = adjusted_2d[0]
        new_target[1] = adjusted_2d[1]
        new_target[2] = self.target_altitude
        
        #clamp to workspace bounds
        new_target[0] = np.clip(new_target[0], self.workspace_bounds[0][0], self.workspace_bounds[0][1])
        new_target[1] = np.clip(new_target[1], self.workspace_bounds[1][0], self.workspace_bounds[1][1])
        
        #debug output
        if self.debug and self.debug_counter % 48 == 0:
            print(f"\n[DEBUG] Current: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            print(f"[DEBUG] Gradient: [{gradient[0]:.4f}, {gradient[1]:.4f}] (mag: {gradient_magnitude:.4f})")
            print(f"[DEBUG] Potential: {current_potential:.4f} | Goal dist: {current_distance_to_goal:.3f}m")
            print(f"[DEBUG] Stuck: {self.stuck_counter} | Circ.sample: {use_circular_sampling}")
            print(f"[DEBUG] Vel mag: {velocity_magnitude:.3f} m/s")
            print(f"[DEBUG] Target: [{new_target[0]:.3f}, {new_target[1]:.3f}, {new_target[2]:.3f}]")
        
        self.debug_counter += 1
        
        return new_target
    
    def check_goal_reached(self, position: np.ndarray) -> bool:
        distance_to_goal = np.linalg.norm(position - self.goal_pos)
        return distance_to_goal < self.goal_tolerance
    
    def run_navigation(self, use_subharmonic: bool = True, visualize_field: bool = True):
        #solve the potential field
        print("Solving potential field")
        iterations = self.field_engine.solve_harmonic_field(verbose=True)
        
        if use_subharmonic:
            print("Computing subharmonic field")
            self.field_engine.solve_subharmonic_field(
                a_att=0.01,
                a_rep=1.0,
                k_rep=1.0,
                verbose=True
            )
        
        print("Starting navigation")
        print(f"Start: {self.start_pos}")
        print(f"Goal: {self.goal_pos}")
        
        #test gradient at start
        test_grad = self.field_engine.compute_gradient(self.start_pos[:2])
        test_pot = self.field_engine.compute_potential(self.start_pos[:2])
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
                print(f"  Distance to goal: {np.linalg.norm(current_pos - self.goal_pos):.3f}m")
                break
            
            #compute next target position from potential field
            self.target_position = self.compute_target_position(current_pos)
            
            #use PID controller to compute motor RPMs from target position
            #computeControlFromState returns (rpm, pos_error, yaw_error)
            target_rpy = np.array([0, 0, 0])  #keep level
            rpm, _, _ = self.ctrl.computeControlFromState(
                control_timestep=self.dt,
                state=state,
                target_pos=self.target_position,
                target_rpy=target_rpy
            )
            
            #apply RPM action (this is what CtrlAviary expects: 4 RPM values)
            obs, reward, terminated, truncated, info = self.env.step(rpm.reshape(1, 4))
            
            #update time
            self.current_time += self.dt
            step_count += 1
            
            #print progress
            if step_count % (2 * self.env.CTRL_FREQ) == 0:
                dist_to_goal = np.linalg.norm(current_pos - self.goal_pos)
                potential = self.field_engine.compute_potential(current_pos[:2])
                dist_to_target = np.linalg.norm(current_pos - self.target_position)
                print(f"t={self.current_time:.1f}s | pos=[{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] | "
                      f"goal_dist={dist_to_goal:.2f}m | phi={potential:.3f} | tracking_err={dist_to_target:.3f}m")
            
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
                
                #check progress
                distances = [np.linalg.norm(pos - self.goal_pos) for pos in self.trajectory]
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
        
        return self.reached_goal, np.array(self.trajectory) if len(self.trajectory) > 0 else np.array([])
    
    def visualize_results(self):

        if len(self.trajectory) == 0:
            print("No trajectory to visualize!")
            return
        
        fig = self.field_engine.visualize(
            show_gradient=True,
            gradient_skip=5,
            agent_position=self.trajectory[-1][:2],
            show_temporal_derivative=False
        )
        
        #overlay trajectory
        ax = fig.axes[0]
        trajectory_2d = np.array([pos[:2] for pos in self.trajectory])
        
        ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 
               'b-', linewidth=2, label='Trajectory', alpha=0.8)
        ax.plot(self.start_pos[0], self.start_pos[1], 
               'go', markersize=15, label='Start', markeredgecolor='white', markeredgewidth=2)
        ax.plot(trajectory_2d[-1, 0], trajectory_2d[-1, 1], 
               'bs', markersize=12, label='End', markeredgecolor='white', markeredgewidth=2)
        
        ax.legend(loc='upper right')
        ax.set_title(f'Navigation {"SUCCESS!" if self.reached_goal else "Incomplete"}')
        
        plt.tight_layout()
        plt.show()
        
        #3D plot
        if len(self.trajectory) > 1:
            self.plot_3d_trajectory()
    
    def plot_3d_trajectory(self):

        trajectory_array = np.array(self.trajectory)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        #trajectory
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2],
               'b-', linewidth=2, label='Trajectory')
        
        #start and goal
        ax.scatter(self.start_pos[0], self.start_pos[1], self.start_pos[2],
                  c='green', s=200, marker='o', label='Start', edgecolors='black', linewidths=2)
        ax.scatter(self.goal_pos[0], self.goal_pos[1], self.goal_pos[2],
                  c='red', s=200, marker='*', label='Goal', edgecolors='black', linewidths=2)
        
        #obstacles
        for obstacle in self.field_engine.obstacles:
            theta = np.linspace(0, 2*np.pi, 30)
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


def example_simple_navigation():
    """
    Example: Simple navigation.
    """
    print("\n" + "="*60)
    print("Potential Field Navigation with PID Control")
    print("="*60 + "\n")
    
    navigator = PotentialFieldNavigator(
        start_pos=np.array([1.0, 1.0, 0.5]),
        goal_pos=np.array([8.0, 8.0, 1.0]),
        workspace_bounds=((0, 10), (0, 10)),
        target_altitude=1.0,
        grid_resolution=100,
        gui=True,
        record_video=False
    )
    
    success, trajectory = navigator.run_navigation(
        use_subharmonic=True,
        visualize_field=True
    )
    
    return navigator, success


def example_obstacle_avoidance():
    """
    Example: Obstacle avoidance.
    """
    print("\n" + "="*60)
    print("Obstacle Avoidance Navigation")
    print("="*60 + "\n")
    
    navigator = PotentialFieldNavigator(
        start_pos=np.array([1.0, 1.0, 0.5]),
        goal_pos=np.array([9.0, 5.0, 1.0]),
        workspace_bounds=((0, 10), (0, 10)),
        target_altitude=1.0,
        grid_resolution=150,
        gui=True,
        record_video=False
    )
    
    navigator.add_cylindrical_obstacle(center_xy=np.array([3.5, 5.0]), radius=0.6)
    navigator.add_cylindrical_obstacle(center_xy=np.array([6.5, 5.0]), radius=0.6)
    navigator.add_cylindrical_obstacle(center_xy=np.array([9.0, 2.0]), radius=0.6)
    
    success, trajectory = navigator.run_navigation(
        use_subharmonic=True,
        visualize_field=True
    )
    
    return navigator, success


if __name__ == "__main__":
    #run simple navigation first
    # navigator, success = example_simple_navigation()
    
    #if successful, try obstacle avoidance
    navigator, success = example_obstacle_avoidance()
    
    print("\nDone!")