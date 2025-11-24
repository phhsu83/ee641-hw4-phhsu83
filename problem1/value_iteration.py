"""
Value Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class ValueIteration:
    """
    Value Iteration solver for gridworld MDP.

    Computes optimal value function V* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Value Iteration solver.

        Args:
            env: GridWorld environment
            gamma: Discount factor
            epsilon: Convergence threshold
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = env.grid_size ** 2
        self.n_actions = env.action_space

    def solve(self, max_iterations: int = 1000) -> Tuple[np.ndarray, int]:
        """
        Run value iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            values: Converged value function V(s)
            n_iterations: Number of iterations until convergence
        """
        # TODO: Initialize value function to zeros
        # TODO: Iterate until convergence:
        #       - For each state:
        #           - Compute Q(s,a) for all actions using Bellman backup
        #           - Set V(s) = max_a Q(s,a)
        #       - Check convergence: max|V_new - V_old| < epsilon
        #       - Update value function
        # TODO: Return final values and iteration count
        # 初始化 V(s) = 0
        values = np.zeros(self.n_states, dtype=float)

        for it in range(max_iterations):
            new_values = np.copy(values)
            delta = 0.0  # 用來檢查收斂：max |V_new - V_old|

            for s in range(self.n_states):
                # 終止狀態的 V(s) 通常設為 0（或固定值，依作業定義）
                if self.env.is_terminal(s):
                    new_values[s] = 0.0
                else:
                    # 對此 state 做一次 Bellman backup
                    new_values[s] = self.bellman_backup(s, values)

                delta = max(delta, abs(new_values[s] - values[s]))

            values = new_values

            # 收斂判斷
            if delta < self.epsilon:
                return values, it + 1

        # 若跑滿 max_iterations 仍未收斂
        return values, max_iterations

        # raise NotImplementedError

    def compute_q_values(self, state: int, values: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all actions in a state.

        Args:
            state: State index
            values: Current value function

        Returns:
            q_values: Array of Q(s,a) for each action
        """
        # TODO: For each action:
        #       - Get transition probabilities P(s'|s,a)
        #       - Compute expected value:
        #           Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
        # TODO: Return Q-values array
        q_values = np.zeros(self.n_actions, dtype=float)

        # 終止狀態：所有 Q(s,a) = 0
        if self.env.is_terminal(state):
            return q_values

        for a in range(self.n_actions):
            # P(s'|s,a) 是一個 dict: next_state -> prob
            trans_prob = self.env.get_transition_prob(state, a)

            q = 0.0
            for next_state, p in trans_prob.items():
                r = self.env.get_reward(state, a, next_state)
                q += p * (r + self.gamma * values[next_state])

            q_values[a] = q

        return q_values

        # raise NotImplementedError

    def extract_policy(self, values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from value function.

        Args:
            values: Optimal value function

        Returns:
            policy: Array of optimal actions for each state
        """
        # TODO: For each state:
        #       - Compute Q-values for all actions
        #       - Select action with maximum Q-value
        # TODO: Return policy array
        policy = np.zeros(self.n_states, dtype=int)

        for s in range(self.n_states):
            if self.env.is_terminal(s):
                # 對 terminal state，policy 隨便給一個 action（不會再用到）
                policy[s] = 0
            else:
                q_values = self.compute_q_values(s, values)
                best_action = int(np.argmax(q_values))
                policy[s] = best_action

        return policy

        # raise NotImplementedError

    def bellman_backup(self, state: int, values: np.ndarray) -> float:
        """
        Perform Bellman backup for a single state.

        Args:
            state: State index
            values: Current value function

        Returns:
            Updated value for state
        """
        # TODO: If terminal state, return 0
        # TODO: Compute Q-values for all actions
        # TODO: Return maximum Q-value
        # 終止狀態的值（這裡設為 0）
        if self.env.is_terminal(state):
            return 0.0

        # V(s) = max_a Q(s,a)
        q_values = self.compute_q_values(state, values)
        return float(np.max(q_values))

        # raise NotImplementedError

    def compute_bellman_error(self, values: np.ndarray) -> float:
        """
        Compute Bellman error for current value function.

        Bellman error = max_s |V(s) - max_a Q(s,a)|

        Args:
            values: Current value function

        Returns:
            Maximum Bellman error across all states
        """
        # TODO: For each state:
        #       - Compute optimal value using Bellman backup
        #       - Calculate absolute difference from current value
        # TODO: Return maximum error
        max_error = 0.0

        for s in range(self.n_states):
            target_v = self.bellman_backup(s, values)  # max_a Q(s,a)
            err = abs(values[s] - target_v)
            if err > max_error:
                max_error = err

        return max_error

        # raise NotImplementedError