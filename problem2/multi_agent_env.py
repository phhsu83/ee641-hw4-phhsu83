"""
Multi-agent gridworld environment with partial observations and communication.
"""

import numpy as np
from typing import Tuple, Optional, List


class MultiAgentEnv:
    """
    Two-agent cooperative gridworld with partial observations.

    Agents must coordinate to simultaneously reach a target cell.
    Each agent observes a 3x3 local patch and exchanges communication signals.
    """

    def __init__(self, grid_size: Tuple[int, int] = (10, 10), obs_window: int = 3,
                 max_steps: int = 50, seed: Optional[int] = None):
        """
        Initialize multi-agent environment.

        Args:
            grid_size: Tuple defining grid dimensions (default 10x10)
            obs_window: Size of local observation window (must be odd, default 3)
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.obs_window = obs_window
        self.max_steps = max_steps

        if seed is not None:
            np.random.seed(seed)

        # Initialize grid components
        self._initialize_grid()

        # Agent state
        self.agent_positions = [None, None]
        self.comm_signals = [0.0, 0.0]
        self.step_count = 0

    def _initialize_grid(self) -> None:
        """
        Create grid with obstacles and target.

        Grid values:
        - 0: Free cell
        - 1: Obstacle
        - 2: Target
        """
        # TODO: Create empty grid of size grid_size
        # TODO: Randomly place up to 6 obstacles (avoiding corners)
        # TODO: Randomly place exactly 1 target cell
        # TODO: Store grid as self.grid
        rows, cols = self.grid_size

        # Create empty grid
        self.grid = np.zeros((rows, cols), dtype=int)

        # Randomly place up to 6 obstacles (avoiding corners)
        corners = {(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)}
        all_cells = [(r, c) for r in range(rows) for c in range(cols)
                     if (r, c) not in corners]

        num_obstacles = np.random.randint(0, 7)  # 0~6
        if num_obstacles > 0 and len(all_cells) > 0:
            obstacle_indices = np.random.choice(len(all_cells), size=min(num_obstacles, len(all_cells)),
                                                replace=False)
            for idx in obstacle_indices:
                r, c = all_cells[idx]
                self.grid[r, c] = 1  # obstacle

        # Randomly place exactly 1 target cell on a non-obstacle cell
        free_positions = list(zip(*np.where(self.grid == 0)))
        if not free_positions:
            # 極端情況：萬一全被障礙塞滿，就清空重來
            self.grid[:, :] = 0
            free_positions = list(zip(*np.where(self.grid == 0)))

        target_pos = free_positions[np.random.randint(len(free_positions))]
        self.grid[target_pos] = 2  # target

        self.target_pos = target_pos

        # raise NotImplementedError

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment to initial state.

        Returns:
            obs_A: Observation for Agent A (11-dimensional vector)
            obs_B: Observation for Agent B (11-dimensional vector)

        Observation format:
        - Elements 0-8: Flattened 3x3 grid patch (row-major order)
        - Element 9: Communication signal from other agent
        - Element 10: Normalized L2 distance between agents
        """
        # TODO: Reset step counter
        # TODO: Randomly place both agents on free cells (not obstacles or target)
        # TODO: Initialize communication signals to 0.0
        # TODO: Generate observations for both agents
        # TODO: Return (obs_A, obs_B)
        self.step_count = 0

        # Find free cells (not obstacles, not target)
        free_cells = self._find_free_cells()
        if len(free_cells) < 2:
            raise RuntimeError("Not enough free cells to place both agents.")

        # Randomly place both agents
        chosen_idx = np.random.choice(len(free_cells), size=2, replace=False)
        pos_A = free_cells[chosen_idx[0]]
        pos_B = free_cells[chosen_idx[1]]
        self.agent_positions = [pos_A, pos_B]

        # Initialize communication signals
        self.comm_signals = [0.0, 0.0]

        # Generate observations
        obs_A = self._get_observation(0)
        obs_B = self._get_observation(1)

        # Append normalized L2 distance between agents
        dist = self._agent_distance()
        obs_A = np.concatenate([obs_A, np.array([dist], dtype=float)])
        obs_B = np.concatenate([obs_B, np.array([dist], dtype=float)])

        return obs_A, obs_B

        # raise NotImplementedError

    def _agent_distance(self) -> float:
        """
        Compute normalized L2 distance between agents.
        """
        rows, cols = self.grid_size
        pos_A, pos_B = self.agent_positions
        dr = pos_A[0] - pos_B[0]
        dc = pos_A[1] - pos_B[1]
        dist = np.sqrt(dr ** 2 + dc ** 2)
        if np.sqrt(rows ** 2 + cols ** 2) > 0:
            return float(dist / np.sqrt(rows ** 2 + cols ** 2))
        
        return 0.0
    

    def step(self, action_A: int, action_B: int, comm_A: float, comm_B: float) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], float, bool]:
        """
        Execute one environment step.

        Args:
            action_A: Agent A's movement action (0:Up, 1:Down, 2:Left, 3:Right, 4:Stay)
            action_B: Agent B's movement action
            comm_A: Communication signal from Agent A to B
            comm_B: Communication signal from Agent B to A

        Returns:
            observations: Tuple of (obs_A, obs_B), each 11-dimensional
            reward: +10 if both agents at target, +2 if one agent at target, -0.1 per step
            done: True if both agents at target or max steps reached
        """
        # TODO: Update agent positions based on actions
        #       - Check boundaries and obstacles
        #       - Invalid moves result in no position change
        # TODO: Store new communication signals for next observation
        # TODO: Check reward condition (both agents at target)
        # TODO: Update step count and check termination
        # TODO: Generate new observations with updated comm signals
        # TODO: Return ((obs_A, obs_B), reward, done)
        # Update agent positions based on actions
        pos_A = self._apply_action(self.agent_positions[0], action_A)
        pos_B = self._apply_action(self.agent_positions[1], action_B)
        self.agent_positions = [pos_A, pos_B]

        # Store new communication signals for next observation
        self.comm_signals = [float(comm_A), float(comm_B)]

        # Compute reward
        at_target_A = (self.grid[pos_A] == 2)
        at_target_B = (self.grid[pos_B] == 2)

        
        reward = -1  # step penalty
        if at_target_A and at_target_B:
            reward += 10.0
        elif at_target_A or at_target_B:
            reward += 0.5
        '''
        # 建議 shaping
        reward = -0.05  # 減少步懲罰，避免過度趕路
        dist_agents = self._agent_distance()  # [0, 1]
        reward += 0.2 * (1.0 - dist_agents)   # 彼此靠近加分

        if at_target_A and at_target_B:
            reward += 10.0
            # 若同一步都從非 T 進入 T，可再加一些
        elif at_target_A or at_target_B:
            reward += 0.5  # 降低單人到達誘因
        '''

        # Update step count and check termination
        self.step_count += 1
        done = False
        if at_target_A and at_target_B:
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        # Generate new observations with updated comm signals and distance
        obs_A = self._get_observation(0)
        obs_B = self._get_observation(1)
        dist = self._agent_distance()
        obs_A = np.concatenate([obs_A, np.array([dist], dtype=float)])
        obs_B = np.concatenate([obs_B, np.array([dist], dtype=float)])

        return (obs_A, obs_B), reward, done

        # raise NotImplementedError

    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """
        Extract local observation for an agent.

        Args:
            agent_idx: Agent index (0 for A, 1 for B)

        Returns:
            observation: 10-dimensional vector
        """
        # TODO: Get agent position
        # TODO: Extract 3x3 patch centered on agent
        #       - Cells outside grid should be -1
        #       - Use grid values (0: free, 1: obstacle, 2: target)
        # TODO: Flatten patch to 9 elements
        # TODO: Append communication signal from other agent
        # TODO: Return 10-dimensional observation
        pos = self.agent_positions[agent_idx]
        rows, cols = self.grid_size
        w = self.obs_window
        half = w // 2

        # Initialize patch with -1 (out-of-bounds)
        patch = np.full((w, w), -1, dtype=float)

        r0, c0 = pos
        for i in range(w):
            for j in range(w):
                rr = r0 + i - half
                cc = c0 + j - half
                if 0 <= rr < rows and 0 <= cc < cols:
                    patch[i, j] = float(self.grid[rr, cc])

        patch_flat = patch.flatten()

        # Communication signal from the other agent
        other_idx = 1 - agent_idx
        comm_from_other = float(self.comm_signals[other_idx])

        observation = np.concatenate([patch_flat, np.array([comm_from_other], dtype=float)])
        
        return observation  # length = 9 + 1 = 10

        # raise NotImplementedError

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        # TODO: Check if position is within grid bounds
        # TODO: Check if position is not an obstacle (grid value != 1)
        r, c = pos
        rows, cols = self.grid_size

        if not (0 <= r < rows and 0 <= c < cols):
            return False

        # Not an obstacle
        if self.grid[r, c] == 1:
            return False

        return True

        # raise NotImplementedError

    def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Apply movement action to position.

        Args:
            pos: Current position (row, col)
            action: Movement action (0-4)

        Returns:
            new_pos: Updated position (stays same if invalid)
        """
        # TODO: Map action to position delta
        #       0: Up (-1, 0)
        #       1: Down (+1, 0)
        #       2: Left (0, -1)
        #       3: Right (0, +1)
        #       4: Stay (0, 0)
        # TODO: Calculate new position
        # TODO: Return new position if valid, else return original position
        # Map action to position delta
        if action == 0:      # Up
            delta = (-1, 0)
        elif action == 1:    # Down
            delta = (1, 0)
        elif action == 2:    # Left
            delta = (0, -1)
        elif action == 3:    # Right
            delta = (0, 1)
        elif action == 4:    # Stay
            delta = (0, 0)
        else:
            # 無效 action，視為 Stay
            delta = (0, 0)

        new_pos = (pos[0] + delta[0], pos[1] + delta[1])

        # Return new position if valid, else original position
        if self._is_valid_position(new_pos):
            return new_pos
        
        return pos

        # raise NotImplementedError

    def _find_free_cells(self) -> List[Tuple[int, int]]:
        """
        Find all free cells in the grid.

        Returns:
            List of (row, col) positions that are free
        """
        # TODO: Iterate through grid
        # TODO: Collect positions where grid value is 0 (free)
        # TODO: Return list of free positions
        rows, cols = self.grid.shape
        free_positions: List[Tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                if self.grid[r, c] == 0:  # free
                    free_positions.append((r, c))
        
        return free_positions

        # raise NotImplementedError

    def render(self) -> None:
        """
        Render current environment state.
        """
        # TODO: Create visual representation of grid
        # TODO: Show agent positions (A, B)
        # TODO: Show target (T)
        # TODO: Show obstacles (X)
        # TODO: Display current communication values
        rows, cols = self.grid.shape

        # Base characters from grid
        char_grid = np.full((rows, cols), '.', dtype='<U2')
        for r in range(rows):
            for c in range(cols):
                if self.grid[r, c] == 1:
                    char_grid[r, c] = 'X'  # obstacle
                elif self.grid[r, c] == 2:
                    char_grid[r, c] = 'T'  # target

        # Place agents
        pos_A, pos_B = self.agent_positions
        if pos_A == pos_B:
            # Both agents on same cell
            r, c = pos_A
            char_grid[r, c] = '*'
        else:
            rA, cA = pos_A
            rB, cB = pos_B
            char_grid[rA, cA] = 'A'
            char_grid[rB, cB] = 'B'

        # Print grid
        print(f"Step: {self.step_count}")
        for r in range(rows):
            print(' '.join(char_grid[r, :]))

        print(f"Target position: {self.target_pos}")
        print(f"Comm A→B: {self.comm_signals[0]:.3f}, B→A: {self.comm_signals[1]:.3f}")
        print()

        # raise NotImplementedError