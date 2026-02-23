"""
Potential Field Engine for Drone (UAV) Navigation

This is the implementation time-varying subharmonic artificial potential fields (TV-SAPF)
for UAV obstacle avoidance and navigation. 

The mathematics are based on:

1. Harmonic Potential Fields: ∇²Φ = 0 (Laplace's equation)
   - Guarantee no local minima in the interior
   - But suffer from the "flat region problem" far from goals

2. Subharmonic Potential Fields: ΔΦ ≥ 0 
   - Maintain the no-local-minima property
   - Amplify gradients near the goal region to address flat region problem

3. Time-Varying Fields: Φ(x, t)
   - Adapt to moving obstacles - with incorporation for later - i am just trying to make it work first and see how much time it takes to run
   - Temporal derivative ∂Φ/∂t provides predictive information
"""

import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib import cm
from dataclasses import dataclass


@dataclass
class Obstacle:
    """
    Represents an obstacle in the environment - maybe will make this be a drone later
    """
    position: np.ndarray  # [x, y] position
    radius: float  # collision radius
    velocity: np.ndarray = None  # [vx, vy] for moving obstacles
    
    def update_position(self, dt: float):
        """
        Update obstacle position based on velocity
        Parameters:
        - dt: time step
        Returns: none 
        """
        if self.velocity is not None:
            self.position = self.position + self.velocity * dt


class PotentialFieldEngine2D:
    """
    2D Potential Field Engine implements harmonic and subharmonic fields.
    This uses finite difference methods on a discretized grid to solve the boundary value problem. 
    
    Why the grid approach?
    - Easier to handle complex boundary conditions (arbitrary obstacles)
    - Compute derivatives numerically
    - Update fields as obstacles move

    I am working towards visualization of the potential field, gradients and temporal derivatives to understand how the field evolves over time.
    """
    
    def __init__(self, x_bounds: tuple[float, float], y_bounds: tuple[float, float], grid_resolution: int = 100, goal_position: np.ndarray = None,
                 max_iterations: int = 1000, convergence_threshold: float = 1e-4)->None:
        """
        Parameters:
        - x_bounds : tuple - (x_min, x_max) bounds of the workspace
        - y_bounds : tuple - (y_min, y_max) bounds of the workspace
        - grid_resolution : int - nr of grid points along each dimension
        - goal_position : np.ndarray - goal position [x, y]
        - max_iterations : int - maximum iterations for the iterative solver
        - convergence_threshold : float - convergence criterion for the solver
        """
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.grid_resolution = grid_resolution
        
        #create the spatial grid
        #discretize the continuous workspace into a grid of points
        self.x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
        self.y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
        self.dx = self.x[1] - self.x[0]  #grid spacing
        self.dy = self.y[1] - self.y[0]  #grid spacing
        
        #create meshgrid for visualization and computation
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        #the potential field Φ(x, y, t)
        self.phi = np.zeros((grid_resolution, grid_resolution))
        self.phi_prev = np.zeros_like(self.phi)  #the previous for computing ∂Φ/∂t
        
        #goal position
        self.goal_position = goal_position if goal_position is not None else \
                            np.array([(x_bounds[0] + x_bounds[1])/2, 
                                     (y_bounds[0] + y_bounds[1])/2])
        
        #obstacles list
        self.obstacles: List[Obstacle] = []
        
        #solver parameters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        #time tracking for temporal derivatives
        self.current_time = 0.0
        self.dt = 0.1  #default timestep

        #subharmonic mode flag (analytical vs grid-based)
        self.subharmonic_mode = False
        self.a_att = 0.01
        self.a_rep = 1.0
        self.k_rep = 1.0
        self.danger_distance = 1.0
        
    def add_obstacle(self, position: np.ndarray, radius: float, 
                     velocity: np.ndarray = None)->None:
        """
        Adds an obstacle to the environment
        
        Parameters:
        - position : np.ndarray - [x, y] position of obstacle center
        - radius : float - collision radius of the obstacle
        - velocity : np.ndarray, optional - [vx, vy] velocity for moving obstacles
        Returns: none
        """
        if velocity is not None:
            obstacle = Obstacle(
            position=np.array(position, dtype=float),
            radius=radius,
            velocity=np.array(velocity, dtype=float)
        )
        else:
            obstacle = Obstacle(
                position=np.array(position, dtype=float),
                radius=radius,
                velocity=None
            )
        self.obstacles.append(obstacle)
        
    def clear_obstacles(self):
        """
        Removes all obstacles from the environment
        Parameters: none
        Returns: none
        """
        self.obstacles.clear()
        
    def _world_to_grid(self, position: np.ndarray) -> tuple[int, int]:
        """
        Converts world coordinates to grid indices
        This handles the mapping between continuous world space and discrete grid space
        Parameters:
        - position : np.ndarray - [x, y] position in world coordinates
        Returns:
        - (i, j) : tuple - grid indices corresponding to the position
        """
        i = int((position[0] - self.x_bounds[0]) / self.dx)
        j = int((position[1] - self.y_bounds[0]) / self.dy)
        
        #clamp to grid boundaries
        i = np.clip(i, 0, self.grid_resolution - 1)
        j = np.clip(j, 0, self.grid_resolution - 1)
        
        return i, j
    
    def _grid_to_world(self, i: int, j: int) -> np.ndarray:
        """
        Convert grid indices to world coordinates (reverse of _world_to_grid)
        Parameters:
        - i : int - grid index along x
        - j : int - grid index along y
        Returns:
        - position : np.ndarray - [x, y] position in world coordinates
        """
        x = self.x_bounds[0] + i * self.dx
        y = self.y_bounds[0] + j * self.dy
        return np.array([x, y])
    
    def _create_boundary_conditions(self) -> np.ndarray:
        """
        Creates the boundary condition mask by marking goal and obstacle regions
        
        In potential field methods:
        - Φ = 0 at the goal (minimum potential)
        - Φ = 1 at obstacles (maximum potential)
        - Free space has values that satisfy the field equation
        
        Returns a mask where:
        - 0 = free space (potential will be computed)
        - 1 = goal region (Φ = 0 enforced)
        - 2 = obstacle region (Φ = 1 enforced)

        No parameters
        Returns:
        - mask : np.ndarray - grid of boundary condition types
        """
        mask = np.zeros((self.grid_resolution, self.grid_resolution), dtype=int)
        
        #mark goal region
        goal_i, goal_j = self._world_to_grid(self.goal_position)
        goal_radius_cells = max(2, int(0.1 / self.dx))  #goal region with approx 0.1m radius
        
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                dist_to_goal = np.sqrt((i - goal_i)**2 + (j - goal_j)**2)
                if dist_to_goal <= goal_radius_cells:
                    mask[i, j] = 1  #goal region
        
        #mark obstacle regions
        for obstacle in self.obstacles:
            obs_i, obs_j = self._world_to_grid(obstacle.position)
            radius_cells = int(obstacle.radius / self.dx)
            
            for i in range(self.grid_resolution):
                for j in range(self.grid_resolution):
                    #check if this grid point is inside the obstacle
                    grid_pos = self._grid_to_world(i, j)
                    dist_to_obs = np.linalg.norm(grid_pos - obstacle.position)
                    
                    if dist_to_obs <= obstacle.radius:
                        mask[i, j] = 2  #obstacle region
        
        return mask
    
    def solve_harmonic_field(self, verbose: bool = False) -> int:
        """
        I AM A LEGEND IF I MAKE THIS WORK :,]

        Solves for the harmonic potential field using Gauss-Seidel iteration
        Idea: - implements an iterative solution to Laplace's equation ∇²Φ = 0
        - The discrete Laplacian on a grid is:
        
        ∇²Φ[i,j] ≈ (Φ[i+1,j] + Φ[i-1,j] + Φ[i,j+1] + Φ[i,j-1] - 4*Φ[i,j]) / h²
        
        - For Laplace's equation, this equals zero, so:
        Φ[i,j] = (Φ[i+1,j] + Φ[i-1,j] + Φ[i,j+1] + Φ[i,j-1]) / 4
        
        This is using the "averaging" property of harmonic functions - the value at each point equals the average of its neighbors.
        
        Returns:
        - iterations : int - nr of iterations performed
        """
        #store previous field for temporal derivative computation
        self.phi_prev = self.phi.copy()
        
        #create boundary condition mask
        mask = self._create_boundary_conditions()
        
        #initialize field
        #start with default values: goal = 0, obstacles = 1, free space = 0.5
        self.phi = np.ones((self.grid_resolution, self.grid_resolution)) * 0.5
        
        # Set workspace boundaries to high potential (Dirichlet boundary condition)
        # This treats workspace edges as "soft repulsive walls"
        self.phi[0, :] = 0.9   # left edge
        self.phi[-1, :] = 0.9  # right edge
        self.phi[:, 0] = 0.9   # bottom edge
        self.phi[:, -1] = 0.9  # top edge
        
        self.phi[mask == 1] = 0.0  #goal (lowest potential)
        self.phi[mask == 2] = 1.0  #obstacles (highest potential)
        
        # Gauss-Seidel iteration
        #this is an iterative method that updates each point based on its neighbors
        #the updates happen in-place, which (is supposed to) speeds up convergence
        for iteration in range(self.max_iterations):
            phi_old = self.phi.copy()
            #update interior points
            for i in range(1, self.grid_resolution - 1):
                for j in range(1, self.grid_resolution - 1):
                    #skip boundary conditions (goal and obstacles)
                    if mask[i, j] != 0:
                        continue
                    # harmonic update: average of neighbors
                    self.phi[i, j] = 0.25 * (
                        self.phi[i+1, j] + self.phi[i-1, j] +
                        self.phi[i, j+1] + self.phi[i, j-1]
                    )
            
            # Keep workspace boundaries at high potential (Dirichlet BC)
            # This creates gentle repulsion from edges
            self.phi[0, :] = 0.9
            self.phi[-1, :] = 0.9
            self.phi[:, 0] = 0.9
            self.phi[:, -1] = 0.9
            
            #re-enforce boundary conditions - cause this is giving me issues (mental)
            self.phi[mask == 1] = 0.0
            self.phi[mask == 2] = 1.0
            
            #check convergence
            max_change = np.max(np.abs(self.phi - phi_old))
            if max_change < self.convergence_threshold:
                if verbose:
                    print(f"Harmonic field converged in {iteration + 1} iterations")
                return iteration + 1
        
        if verbose:
            print(f"Harmonic field reached max iterations ({self.max_iterations}) without full convergence")
        return self.max_iterations
    
    def solve_subharmonic_field(self, a_att: float = 0.01, a_rep: float = 1.0,
                                k_rep: float = 1.0, verbose: bool = False) -> None:
        """
        Computes the subharmonic potential field using analytical functions
        from arXiv:2402.11601.

        Complete potential field (based on arXiv:2402.11601):
            phi(x,y) = a_att * r_goal^2 + sum( a_rep * exp(-k * r_obs_i^2) )

        where:
            r_goal^2 = (x - xg)^2 + (y - yg)^2   (distance to goal squared)
            r_obs_i^2 = (x - xoi)^2 + (y - yoi)^2 (distance to obstacle i squared)

        Attractive field: phi_att = a_att * r^2
            Creates a bowl with minimum at goal. Gradient descent pulls toward goal.

        Repulsive field: phi_rep = +a * exp(-k * r^2)
            Creates positive barriers (peaks) at obstacles.
            -gradient pushes drone away from obstacles.
            danger distance d0 = sqrt(1/k)  (Eq. 20)

        Parameters:
        - a_att : float - attractive field gain (Eq. 21)
        - a_rep : float - repulsive field amplitude (Eq. 13)
        - k_rep : float - repulsive field steepness; danger distance = sqrt(1/k)
        - verbose : bool - print field info
        Returns: none
        """
        #store parameters for analytical gradient/potential computation
        self.subharmonic_mode = True
        self.a_att = a_att
        self.a_rep = a_rep
        self.k_rep = k_rep
        self.danger_distance = np.sqrt(1.0 / k_rep) if k_rep > 0 else 0.0

        #precompute the field on the grid for visualization
        self.phi_prev = self.phi.copy()

        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                grid_pos = self._grid_to_world(i, j)
                self.phi[i, j] = self._analytical_potential(grid_pos)

        if verbose:
            print(f"Subharmonic field computed (analytical)")
            print(f"  a_att={a_att}, a_rep={a_rep}, k_rep={k_rep}")
            print(f"  Danger distance d0 = {self.danger_distance:.3f}m")
            print(f"  Potential range: [{np.min(self.phi):.4f}, {np.max(self.phi):.4f}]")

    def _analytical_potential(self, position: np.ndarray) -> float:
        """
        Computes the analytical subharmonic potential at a position.
        phi(x,y) = a_att * r_goal^2 + sum( a_rep * exp(-k * r_obs_i^2) )

        Attractive term creates a bowl centered at goal (minimum at goal).
        Repulsive term creates positive barriers at obstacles (maxima).
        Gradient descent on phi drives toward goal and away from obstacles.

        Parameters:
        - position : np.ndarray - [x, y] in world coordinates
        Returns:
        - potential : float
        """
        #attractive component: phi_att = a_att * r_goal^2
        #minimum at goal, increases with distance
        diff_goal = position - self.goal_position
        r_goal_sq = np.dot(diff_goal, diff_goal)
        phi_att = self.a_att * r_goal_sq

        #repulsive component: phi_rep = +a * exp(-k * r_obs^2)
        #positive barrier at each obstacle, -gradient pushes drone away
        phi_rep = 0.0
        for obs in self.obstacles:
            diff_obs = position - obs.position
            r_obs_sq = np.dot(diff_obs, diff_obs)
            phi_rep += self.a_rep * np.exp(-self.k_rep * r_obs_sq)

        return phi_att + phi_rep

    def _analytical_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Computes the analytical gradient of the subharmonic potential.

        grad(phi_att) = 2 * a_att * (x - x_g, y - y_g)   (points away from goal)
        grad(phi_rep) = -2 * a_rep * k * (x - x_o, y - y_o) * exp(-k * r^2)  (points toward obstacle)

        The drone follows -gradient, which moves toward goal and away from obstacles.

        Parameters:
        - position : np.ndarray - [x, y] in world coordinates
        Returns:
        - gradient : np.ndarray - [dphi/dx, dphi/dy]
        """
        #attractive gradient: points away from goal
        #-gradient points toward goal (descent into bowl)
        diff_goal = position - self.goal_position
        grad = 2 * self.a_att * diff_goal

        #repulsive gradient per obstacle
        #derivative of +a*exp(-k*r^2) w.r.t. x is -2*a*k*dx*exp(-k*r^2)
        #this points toward obstacle center (uphill toward peak)
        #so -gradient points AWAY from obstacle (repels)
        for obs in self.obstacles:
            diff_obs = position - obs.position
            r_obs_sq = np.dot(diff_obs, diff_obs)
            exp_term = np.exp(-self.k_rep * r_obs_sq)
            grad -= 2 * self.a_rep * self.k_rep * diff_obs * exp_term

        return grad

    def compute_dphi_dt(self, position: np.ndarray) -> float:
        """
        Analytical partial temporal derivative of the subharmonic potential.

        Since the goal is static and obstacle positions are the only
        time-dependent quantities, we differentiate through the Gaussian
        barriers using the chain rule:

            ∂Φ/∂t = Σ_i  2·k·a_rep · exp(-k·‖x - x_oi‖²)
                         · (x - x_oi) · v_oi

        where v_oi is the velocity of obstacle i.

        Sign convention:
          ∂Φ/∂t > 0  ->  obstacle approaching (potential rising)
          ∂Φ/∂t < 0  ->  obstacle receding   (potential falling)
          ∂Φ/∂t ≈ 0  -> static obstacle or tangential motion

        Requires subharmonic_mode=True (analytical field active).

        Parameters
        - position : np.ndarray  [x, y] in world coordinates

        Returns
        - dphi_dt : float
        """
        if not self.subharmonic_mode:
            return 0.0

        dphi_dt = 0.0
        for obs in self.obstacles:
            if obs.velocity is None or np.linalg.norm(obs.velocity) < 1e-12:
                continue  # static obstacle contributes nothing
            diff = position - obs.position
            r_sq = np.dot(diff, diff)
            exp_term = np.exp(-self.k_rep * r_sq)
            # dot product (x - x_o) · v_o
            dot = np.dot(diff, obs.velocity)
            dphi_dt += 2.0 * self.k_rep * self.a_rep * exp_term * dot

        return dphi_dt

    def circular_sample(self, raw_next_point: np.ndarray, current_pos: np.ndarray,
                        sample_radius: float = 0.5, n_samples: int = 16) -> np.ndarray:
        """
        Circular sampling technique (Section IV of arXiv:2402.11601).

        When the gradient-based next move point is in a flat region, sample
        candidate points on a circle around it, filter out points inside
        the danger distance of any obstacle, then select the point whose
        "next-next move point" is closest to the goal.

        Parameters:
        - raw_next_point : np.ndarray - [x, y] raw APF next move point
        - current_pos : np.ndarray - [x, y] current robot position
        - sample_radius : float - radius of the sampling circle
        - n_samples : int - number of points to sample on the circle
        Returns:
        - best_point : np.ndarray - [x, y] adjusted next move point
        """
        #generate candidate points on a circle centered at raw_next_point
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

        #sort angles so we start from the direction closest to goal (paper's priority)
        goal_dir = np.arctan2(self.goal_position[1] - raw_next_point[1],
                              self.goal_position[0] - raw_next_point[0])
        angles = (angles + goal_dir) % (2 * np.pi)
        angles = np.sort(angles)

        candidates = []
        for angle in angles:
            candidate = np.array([
                raw_next_point[0] + sample_radius * np.cos(angle),
                raw_next_point[1] + sample_radius * np.sin(angle)
            ])

            #check workspace bounds
            if (candidate[0] < self.x_bounds[0] or candidate[0] > self.x_bounds[1] or
                candidate[1] < self.y_bounds[0] or candidate[1] > self.y_bounds[1]):
                continue

            #filter out points within danger distance of any obstacle (Eq. 19, 20)
            safe = True
            for obs in self.obstacles:
                dist_to_obs = np.linalg.norm(candidate - obs.position)
                if dist_to_obs < self.danger_distance:
                    safe = False
                    break

            if safe:
                candidates.append(candidate)

        if len(candidates) == 0:
            #no safe candidate found, return raw point
            return raw_next_point

        #evaluate each candidate: compute "next-next move point" and pick
        #the one that gets closest to the goal (paper's evaluation criterion)
        best_candidate = candidates[0]
        best_distance = float('inf')

        for candidate in candidates:
            #compute gradient at candidate to get "next-next move point"
            grad = self._analytical_gradient(candidate)
            grad_mag = np.linalg.norm(grad)
            if grad_mag > 1e-8:
                next_next = candidate - (grad / grad_mag) * sample_radius * 0.5
            else:
                next_next = candidate

            dist_to_goal = np.linalg.norm(next_next - self.goal_position)
            if dist_to_goal < best_distance:
                best_distance = dist_to_goal
                best_candidate = candidate

        return best_candidate
        
    def compute_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Computes the spatial gradient ∇Φ at a given position.
        In subharmonic mode uses analytical gradient (paper Eq. 23 derivative).
        In harmonic mode uses bilinear interpolation on the grid.
        
        Parameters:
        - position : np.ndarray - [x, y] position in world coordinates 
        Returns:
        - gradient : np.ndarray - [∂Φ/∂x, ∂Φ/∂y] gradient vector
        """
        #use analytical gradient in subharmonic mode
        if self.subharmonic_mode:
            return self._analytical_gradient(position)
        #find grid cell containing this position
        i_float = (position[0] - self.x_bounds[0]) / self.dx
        j_float = (position[1] - self.y_bounds[0]) / self.dy
        
        i = int(i_float)
        j = int(j_float)
        
        #clamp to a valid range
        i = np.clip(i, 0, self.grid_resolution - 2)
        j = np.clip(j, 0, self.grid_resolution - 2)
        
        #bilinear interpolation weights
        wx = i_float - i
        wy = j_float - j
        
        #compute gradients using central differences at grid points
        # ∂Φ/∂x ≈ (Φ[i+1,j] - Φ[i-1,j]) / (2*dx)
        grad_x = np.zeros((2, 2))
        grad_y = np.zeros((2, 2))
        
        for di in range(2):
            for dj in range(2):
                ii = min(i + di, self.grid_resolution - 1)
                jj = min(j + dj, self.grid_resolution - 1)
                
                #central differences + boundary handling
                if ii > 0 and ii < self.grid_resolution - 1:
                    grad_x[di, dj] = (self.phi[ii+1, jj] - self.phi[ii-1, jj]) / (2 * self.dx)
                else:
                    grad_x[di, dj] = 0.0
                
                if jj > 0 and jj < self.grid_resolution - 1:
                    grad_y[di, dj] = (self.phi[ii, jj+1] - self.phi[ii, jj-1]) / (2 * self.dy)
                else:
                    grad_y[di, dj] = 0.0
        
        #bilinear interpolation of gradients
        gx = (1-wx)*(1-wy)*grad_x[0,0] + wx*(1-wy)*grad_x[1,0] + \
             (1-wx)*wy*grad_x[0,1] + wx*wy*grad_x[1,1]
        gy = (1-wx)*(1-wy)*grad_y[0,0] + wx*(1-wy)*grad_y[1,0] + \
             (1-wx)*wy*grad_y[0,1] + wx*wy*grad_y[1,1]
        
        return np.array([gx, gy])
    
    def compute_potential(self, position: np.ndarray) -> float:
        """
        Gets the potential value Φ at a given position.
        In subharmonic mode uses analytical function (paper Eq. 23).
        In harmonic mode uses bilinear interpolation on the grid.
        
        Parameters:
        - position : np.ndarray - [x, y] position in world coordinates  
        Returns:
        - potential : float - interpolated potential value
        """
        #use analytical potential in subharmonic mode
        if self.subharmonic_mode:
            return self._analytical_potential(position)
        i_float = (position[0] - self.x_bounds[0]) / self.dx
        j_float = (position[1] - self.y_bounds[0]) / self.dy
        
        i = int(i_float)
        j = int(j_float)
        
        #clamp to valid range
        i = np.clip(i, 0, self.grid_resolution - 2)
        j = np.clip(j, 0, self.grid_resolution - 2)
        
        #bilinear interpolation
        wx = i_float - i
        wy = j_float - j
        
        phi_interp = (1-wx)*(1-wy)*self.phi[i, j] + \
                     wx*(1-wy)*self.phi[i+1, j] + \
                     (1-wx)*wy*self.phi[i, j+1] + \
                     wx*wy*self.phi[i+1, j+1]
        
        return phi_interp
    
    def compute_temporal_derivative(self) -> np.ndarray:
        """
        Computes the temporal derivative ∂Φ/∂t.
        This is the rate of change of the potential field due to obstacle motion.
        - regions where ∂Φ/∂t > 0 indicate the potential is increasing, likely because an obstacle is approaching.
        
        Returns:
        - dphi_dt : np.ndarray - grid of temporal derivative values
        """
        if self.dt > 0:
            return (self.phi - self.phi_prev) / self.dt
        else:
            return np.zeros_like(self.phi)
    
    def get_temporal_derivative_at_position(self, position: np.ndarray) -> float:
        """
        Gets ∂Φ/∂t at a specific position using bilinear interpolation
        
        Parameters:
        - position : np.ndarray - [x, y] position in world coordinates  
        Returns:
        - dphi_dt : float - temporal derivative at the position
        """
        dphi_dt_grid = self.compute_temporal_derivative()
        
        i_float = (position[0] - self.x_bounds[0]) / self.dx
        j_float = (position[1] - self.y_bounds[0]) / self.dy
        
        i = int(i_float)
        j = int(j_float)
        
        #clamp to valid range
        i = np.clip(i, 0, self.grid_resolution - 2)
        j = np.clip(j, 0, self.grid_resolution - 2)
        
        #bilinear interpolation
        wx = i_float - i
        wy = j_float - j
        
        dphi_dt = (1-wx)*(1-wy)*dphi_dt_grid[i, j] + \
                  wx*(1-wy)*dphi_dt_grid[i+1, j] + \
                  (1-wx)*wy*dphi_dt_grid[i, j+1] + \
                  wx*wy*dphi_dt_grid[i+1, j+1]
        
        return dphi_dt
    
    def update_field_analytical(self, dt: float, a_att: float = 0.01, a_rep: float = 1.0, k_rep: float = 1.0,
                                recompute_grid: bool = False, verbose: bool = False) -> None:
        """
        Field update for dynamic obstacles using only the analytical subharmonic field. 

        By default the visualization grid is not recomputed (O(N^2) per step).
        Call with recompute_grid=True, or call solve_subharmonic_field() once after
        the simulation loop, to get a visualization-ready grid.

        Parameters:
        - dt : float - time step
        - a_att, a_rep, k_rep : subharmonic field coefficients (arXiv:2402.11601)
        - recompute_grid : bool - if True, also recompute phi on the full grid (slow)
        - verbose : bool - print info
        """
        self.dt = dt
        self.current_time += dt

        #save previous grid for temporal derivative computation
        self.phi_prev = self.phi.copy()

        #update obstacle positions based on velocity
        for obstacle in self.obstacles:
            obstacle.update_position(dt)

        #store/update analytical parameters so gradient/potential dispatch works
        self.subharmonic_mode = True
        self.a_att = a_att
        self.a_rep = a_rep
        self.k_rep = k_rep
        self.danger_distance = np.sqrt(1.0 / k_rep) if k_rep > 0 else 0.0

        #optionally recompute the full grid (expensive, for visualization only)
        if recompute_grid:
            for i in range(self.grid_resolution):
                for j in range(self.grid_resolution):
                    grid_pos = self._grid_to_world(i, j)
                    self.phi[i, j] = self._analytical_potential(grid_pos)

    def update_field(self, dt: float, subharmonic: bool = True, a_att: float = 0.01, a_rep: float = 1.0, k_rep: float = 1.0,
                    verbose: bool = False)->None:
        """
        Updates the potential field for the current obstacle configuration.
        Includes full harmonic solve + optional subharmonic overlay.
        
        NOTE: For pure analytical subharmonic mode with dynamic obstacles,
        use update_field_analytical() instead - it is much faster because it
        skips the iterative Gauss-Seidel harmonic solve.
        
        Parameters:
        - dt : float - time step (for computing temporal derivative)
        - subharmonic : bool - whether to use analytical subharmonic field (arXiv:2402.11601)
        - a_att : float - attractive strength coefficient (Eq. 21)
        - a_rep : float - repulsive strength coefficient (Eq. 13)
        - k_rep : float - repulsive decay rate (Eq. 13). Danger distance = 1/sqrt(k_rep)
        - verbose : bool - whether to print convergence information
        Returns: none
        """
        self.dt = dt
        self.current_time += dt
        
        #update obstacle positions
        for obstacle in self.obstacles:
            obstacle.update_position(dt)
        
        #solve for harmonic field with new boundary conditions
        iterations = self.solve_harmonic_field(verbose=verbose)
        
        #add subharmonic field if needed
        if subharmonic:
            self.solve_subharmonic_field(
                a_att=a_att,
                a_rep=a_rep,
                k_rep=k_rep,
                verbose=verbose
            )
    
    def visualize(self, show_gradient: bool = True, gradient_skip: int = 5, agent_position: Optional[np.ndarray] = None,show_temporal_derivative: bool = False,
                  save_path: Optional[str] = None)->plt.Figure:
        """
        THIS IS GENERATED VISUALIZATION CODE
        Make visualizations of the potential field with obstacles, goal, and optionally the agent.
        
        Parameters:
        - show_gradient : bool - whether to show gradient vectors
        - gradient_skip : int - show every Nth gradient vector (for clarity)
        - agent_position : np.ndarray, optional - current agent position to mark on the plot
        - show_temporal_derivative : bool - whether to show ∂Φ/∂t in a second subplot
        - save_path : str, optional - path to save the figure
        """
        if show_temporal_derivative:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        
        #plot 1: potential field with gradients
        im1 = ax1.contourf(self.X, self.Y, self.phi.T, levels=20, cmap='viridis')
        plt.colorbar(im1, ax=ax1, label='Potential Φ')
        
        #add gradient vectors
        if show_gradient:
            #compute gradients on the grid
            grad_x = np.zeros_like(self.phi)
            grad_y = np.zeros_like(self.phi)
            
            for i in range(1, self.grid_resolution - 1):
                for j in range(1, self.grid_resolution - 1):
                    grad_x[i, j] = (self.phi[i+1, j] - self.phi[i-1, j]) / (2 * self.dx)
                    grad_y[i, j] = (self.phi[i, j+1] - self.phi[i, j-1]) / (2 * self.dy)
            
            #plot gradient vectors (negative for descent direction)
            skip = gradient_skip
            ax1.quiver(self.X[::skip, ::skip], self.Y[::skip, ::skip],
                      -grad_x.T[::skip, ::skip], -grad_y.T[::skip, ::skip],
                      color='white', alpha=0.6, scale=20)
        
        #mark goal
        ax1.plot(self.goal_position[0], self.goal_position[1], 
                'g*', markersize=20, label='Goal')
        
        #mark obstacles
        for i, obs in enumerate(self.obstacles):
            circle = plt.Circle(obs.position, obs.radius, 
                              color='red', alpha=0.7, 
                              label='Obstacle' if i == 0 else '')
            ax1.add_patch(circle)
            
            #show velocity vector if moving
            if obs.velocity is not None:
                ax1.arrow(obs.position[0], obs.position[1],
                         obs.velocity[0]*0.5, obs.velocity[1]*0.5,
                         head_width=0.1, head_length=0.1, 
                         fc='red', ec='red', alpha=0.7)
        
        #mark agent if given
        if agent_position is not None:
            ax1.plot(agent_position[0], agent_position[1], 
                    'bo', markersize=10, label='Agent')
        
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title('Potential Field Φ(x, y, t)')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        #plot 2: temporal derivative
        if show_temporal_derivative:
            dphi_dt = self.compute_temporal_derivative()
            im2 = ax2.contourf(self.X, self.Y, dphi_dt.T, levels=20, 
                             cmap='RdBu_r', vmin=-np.max(np.abs(dphi_dt)), 
                             vmax=np.max(np.abs(dphi_dt)))
            plt.colorbar(im2, ax=ax2, label='∂Φ/∂t')
            
            ax2.plot(self.goal_position[0], self.goal_position[1], 
                    'g*', markersize=20)
            
            for obs in self.obstacles:
                circle = plt.Circle(obs.position, obs.radius, 
                                  color='red', alpha=0.5)
                ax2.add_patch(circle)
            
            if agent_position is not None:
                ax2.plot(agent_position[0], agent_position[1], 
                        'bo', markersize=10)
            
            ax2.set_xlabel('x (m)')
            ax2.set_ylabel('y (m)')
            ax2.set_title('Temporal Derivative ∂Φ/∂t')
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig