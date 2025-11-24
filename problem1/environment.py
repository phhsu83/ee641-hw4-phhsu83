"""
Stochastic gridworld environment for reinforcement learning.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class GridWorldEnv:
    """
    5x5 Stochastic GridWorld Environment.

    The agent navigates a grid with stochastic transitions:
    - 0.8 probability of moving in the intended direction
    - 0.1 probability of drifting left (perpendicular)
    - 0.1 probability of drifting right (perpendicular)

    Grid layout:
    - Start: (0, 0)
    - Goal: (4, 4)
    - Obstacles: (2, 2), (1, 3)
    - Penalties: (3, 1), (0, 3)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize gridworld environment.

        Args:
            seed: Random seed for reproducibility
        """
        self.grid_size = 5
        self.max_steps = 50

        # Define special cells
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]

        # Rewards
        self.goal_reward = 10.0
        self.penalty_reward = -5.0
        self.step_cost = -0.1

        # Transition probabilities
        self.prob_intended = 0.8
        self.prob_drift = 0.1

        # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = 4
        self.action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']

        if seed is not None:
            np.random.seed(seed)

        self.reset()

    def reset(self) -> int:
        """
        Reset environment to initial state.

        Returns:
            state: Initial state index
        """
        # TODO: Initialize agent position to start_pos
        # TODO: Reset step counter
        # TODO: Set done flag to False
        # TODO: Return state index (use _pos_to_state)
        self.agent_pos = self.start_pos
        self.steps = 0
        self.done = False

        return self._pos_to_state(self.agent_pos)

        # raise NotImplementedError

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action index (0-3)

        Returns:
            next_state: Next state index
            reward: Reward received
            done: Whether episode terminated
            info: Additional information
        """
        # TODO: Check if episode already done
        # TODO: Get next position based on stochastic transitions
        # TODO: Calculate reward (use _calculate_reward helper)
        # TODO: Update position and step count
        # TODO: Check termination conditions
        # TODO: Return (next_state, reward, done, info)
        # 若 episode 已經結束，就不要再動
        if self.done:
            return self._pos_to_state(self.agent_pos), 0.0, True, {
                "msg": "Episode already done"
            }

        # 取得目前位置
        current_pos = self.agent_pos

        # 根據 stochastic transition 規則，取得所有 (next_pos, prob)
        outcomes = self._get_next_positions(current_pos, action)
        # outcomes: List[(pos, prob)]

        # 依機率抽樣一個 next_pos
        rand = np.random.rand()
        cumulative = 0.0
        next_pos = current_pos  # default，理論上不會用到（因為 prob 應該加總為 1）

        for pos, p in outcomes:
            cumulative += p
            if rand <= cumulative:
                next_pos = pos
                break

        # 計算 reward
        reward = self._calculate_reward(next_pos)

        # 更新 agent 狀態與步數
        self.agent_pos = next_pos
        self.steps += 1

        # 判斷是否終止（到達 goal 或超過最大步數）
        if next_pos == self.goal_pos or self.steps >= self.max_steps:
            self.done = True
        else:
            self.done = False

        # 編碼成 state index 並回傳
        next_state = self._pos_to_state(next_pos)
        info = {
            "position": next_pos,
            "steps": self.steps,
            "action": self.action_names[action],
        }

        return next_state, reward, self.done, info

        # raise NotImplementedError

    def get_transition_prob(self, state: int, action: int) -> Dict[int, float]:
        """
        Get transition probabilities P(s'|s,a).

        Args:
            state: Current state index
            action: Action index

        Returns:
            Dictionary mapping next_state -> probability
        """
        # TODO: Convert state to position
        # TODO: For given action, compute all possible next positions
        #       considering stochastic transitions
        # TODO: Handle boundary and obstacle collisions
        # TODO: Return probability distribution over next states
        # Terminal is absorbing
        if self.is_terminal(state):
            return {state: 1.0}

        pos = self._state_to_pos(state)
        outcomes = self._get_next_positions(pos, action)

        dist: Dict[int, float] = {}
        for next_pos, p in outcomes:
            s_next = self._pos_to_state(next_pos)
            dist[s_next] = dist.get(s_next, 0.0) + p

        # 小心浮點累積誤差（可選）
        total = sum(dist.values())
        if total > 0:
            for k in list(dist.keys()):
                dist[k] = dist[k] / total

        return dist

        # raise NotImplementedError

    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Get reward for transition.

        Args:
            state: Current state index
            action: Action taken
            next_state: Resulting state

        Returns:
            Reward value
        """
        # TODO: Convert next_state to position
        # TODO: Check if goal reached (+10)
        # TODO: Check if penalty cell (-5)
        # TODO: Otherwise return step cost (-0.1)
        pos = self._state_to_pos(next_state)

        if pos == self.goal_pos:
            return self.goal_reward
        elif pos in self.penalties:
            return self.penalty_reward
        else:
            return self.step_cost

        # raise NotImplementedError

    def is_terminal(self, state: int) -> bool:
        """
        Check if state is terminal.

        Args:
            state: State index

        Returns:
            True if terminal state
        """
        # TODO: Convert state to position
        # TODO: Return True if position equals goal_pos
        pos = self._state_to_pos(state)

        return pos == self.goal_pos

        # raise NotImplementedError

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """
        Convert grid position to state index.

        Args:
            pos: (row, col) position

        Returns:
            State index (0-24)
        """
        # TODO: Convert 2D position to 1D state index
        # State = row * grid_size + col
        row, col = pos
        
        return row * self.grid_size + col

        # raise NotImplementedError

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """
        Convert state index to grid position.

        Args:
            state: State index

        Returns:
            (row, col) position
        """
        # TODO: Convert 1D state index to 2D position
        # row = state // grid_size
        # col = state % grid_size
        row = state // self.grid_size
        col = state % self.grid_size
        
        return (row, col)

        # raise NotImplementedError

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        # TODO: Check if position is within grid bounds
        # TODO: Check if position is not an obstacle
        row, col = pos

        in_bounds = (0 <= row < self.grid_size) and (0 <= col < self.grid_size)
        not_obstacle = pos not in self.obstacles

        return in_bounds and not_obstacle

        # raise NotImplementedError

    def _get_next_positions(self, pos: Tuple[int, int], action: int) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get possible next positions and probabilities for stochastic transition.

        Args:
            pos: Current position
            action: Action to take

        Returns:
            List of (next_position, probability) tuples
        """
        # TODO: Define action effects (deltas for UP, RIGHT, DOWN, LEFT)
        # TODO: Get intended direction and perpendicular directions
        # TODO: For each possible outcome (intended, drift left, drift right):
        #       - Calculate next position
        #       - If invalid, stay in current position
        #       - Add (position, probability) to list
        # TODO: Merge probabilities for same positions
        # Absorbing terminal (stay in goal)
        if pos == self.goal_pos:
            return [(pos, 1.0)]

        # 4-neighborhood deltas: UP, RIGHT, DOWN, LEFT
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # Left / right drift mapping for each action
        left_drift  = {0: 3, 1: 0, 2: 1, 3: 2}
        right_drift = {0: 1, 1: 2, 2: 3, 3: 0}

        # (action_index, probability)
        candidates = [
            (action,               self.prob_intended),
            (left_drift[action],   self.prob_drift),
            (right_drift[action],  self.prob_drift),
        ]

        # Accumulate probabilities per resulting position (merge duplicates)
        prob_by_pos: Dict[Tuple[int, int], float] = {}

        r, c = pos
        for a_idx, p in candidates:
            dr, dc = deltas[a_idx]
            nxt = (r + dr, c + dc)

            # If invalid, agent stays
            if not self._is_valid_pos(nxt):
                nxt = pos

            prob_by_pos[nxt] = prob_by_pos.get(nxt, 0.0) + p

        # Convert to list
        return list(prob_by_pos.items())

        # raise NotImplementedError

    def _calculate_reward(self, pos: Tuple[int, int]) -> float:
        """
        Calculate reward for entering a position.

        Args:
            pos: Position entered

        Returns:
            Reward value
        """
        # TODO: Check if position is goal (+10)
        # TODO: Check if position is penalty (-5)
        # TODO: Otherwise return step cost (-0.1)
        if pos == self.goal_pos:
            return self.goal_reward
        elif pos in self.penalties:
            return self.penalty_reward
        else:
            return self.step_cost

        # raise NotImplementedError

    def render(self, value_function: Optional[np.ndarray] = None) -> None:
        """
        Render current state of environment.

        Args:
            value_function: Optional value function to display
        """
        # TODO: Create visual representation of grid
        # TODO: Mark current position, goal, obstacles, penalties
        # TODO: If value_function provided, show as heatmap
        print("\n=== GridWorld ===")

        # 畫地圖本身（位置狀態）
        for r in range(self.grid_size):
            row_symbols = []
            for c in range(self.grid_size):
                pos = (r, c)

                if pos == self.agent_pos:
                    symbol = "A"      # Agent
                elif pos == self.goal_pos:
                    symbol = "G"      # Goal
                elif pos in self.obstacles:
                    symbol = "#"      # Obstacle
                elif pos in self.penalties:
                    symbol = "P"      # Penalty cell
                else:
                    symbol = "."      # Normal cell

                row_symbols.append(symbol)

            print(" ".join(row_symbols))

        # 若有提供 value_function，就順便顯示數值（文字版 heatmap）
        if value_function is not None:
            print("\n=== Value Function (per state) ===")
            # 確保是一維向量，長度 = grid_size^2
            vf = np.asarray(value_function)
            assert vf.shape[0] == self.grid_size * self.grid_size, \
                f"value_function length must be {self.grid_size * self.grid_size}, got {vf.shape[0]}"

            # 轉成 2D，方便閱讀
            vf_2d = vf.reshape(self.grid_size, self.grid_size)

            for r in range(self.grid_size):
                row_vals = []
                for c in range(self.grid_size):
                    v = vf_2d[r, c]
                    row_vals.append(f"{v:6.2f}")  # 固定寬度 & 兩位小數，方便對齊
                print(" ".join(row_vals))

        print()  # 空一行美觀

        # raise NotImplementedError