"""
Q-Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class QIteration:
    """
    Q-Iteration solver for gridworld MDP.

    Computes optimal action-value function Q* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Q-Iteration solver.

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
        Run Q-iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            q_values: Converged Q-function Q(s,a)
            n_iterations: Number of iterations until convergence
        """
        # TODO: Initialize Q-function to zeros (shape: [n_states, n_actions])
        # TODO: Iterate until convergence:
        #       - For each state-action pair:
        #           - Compute updated Q-value using Bellman equation:
        #             Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * max_a' Q(s',a')]
        #       - Check convergence: max|Q_new - Q_old| < epsilon
        #       - Update Q-function
        # TODO: Return final Q-values and iteration count
        # 初始化 Q-function，全 0，shape = [n_states, n_actions]
        q_values = np.zeros((self.n_states, self.n_actions), dtype=float)

        for it in range(max_iterations):
            new_q = np.copy(q_values)
            delta = 0.0  # max |Q_new - Q_old|

            for s in range(self.n_states):
                # 終止狀態：所有 Q(s,a) = 0
                if self.env.is_terminal(s):
                    new_q[s, :] = 0.0
                    continue

                for a in range(self.n_actions):
                    updated_q = self.bellman_update(s, a, q_values)
                    delta = max(delta, abs(updated_q - q_values[s, a]))
                    new_q[s, a] = updated_q

            q_values = new_q

            if delta < self.epsilon:
                return q_values, it + 1

        # 未在 max_iterations 內收斂，就回傳最後的結果
        return q_values, max_iterations

        raise NotImplementedError

    def bellman_update(self, state: int, action: int, q_values: np.ndarray) -> float:
        """
        Compute updated Q-value for a state-action pair.

        Args:
            state: State index
            action: Action index
            q_values: Current Q-function

        Returns:
            Updated Q-value for (s,a)
        """
        # TODO: Get transition probabilities P(s'|s,a)
        # TODO: For each possible next state:
        #       - Get reward R(s,a,s')
        #       - Get max Q-value for next state: max_a' Q(s',a')
        #       - Accumulate: prob * [reward + gamma * max_q_next]
        # TODO: Return updated Q-value
        # 若是終止狀態，Q(s,a) = 0
        if self.env.is_terminal(state):
            return 0.0

        # 取得 P(s'|s,a)
        trans_prob = self.env.get_transition_prob(state, action)

        updated_q = 0.0
        for next_state, p in trans_prob.items():
            reward = self.env.get_reward(state, action, next_state)
            max_q_next = np.max(q_values[next_state])  # max_a' Q(s', a')
            updated_q += p * (reward + self.gamma * max_q_next)

        return updated_q

        raise NotImplementedError

    def extract_policy(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from Q-function.

        Args:
            q_values: Optimal Q-function

        Returns:
            policy: Array of optimal actions for each state
        """
        # TODO: For each state:
        #       - Select action with maximum Q-value: argmax_a Q(s,a)
        # TODO: Return policy array
        policy = np.zeros(self.n_states, dtype=int)

        for s in range(self.n_states):
            if self.env.is_terminal(s):
                policy[s] = 0  # 終止狀態隨便給一個 action，不會實際用到
            else:
                policy[s] = int(np.argmax(q_values[s]))

        return policy

        raise NotImplementedError

    def extract_values(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract value function from Q-function.

        Args:
            q_values: Q-function

        Returns:
            values: State value function V(s) = max_a Q(s,a)
        """
        # TODO: For each state:
        #       - Compute V(s) = max_a Q(s,a)
        # TODO: Return value function
        # 對每個 state 取 over actions 的最大值
        values = np.max(q_values, axis=1)
        
        return values

        raise NotImplementedError

    def compute_bellman_error(self, q_values: np.ndarray) -> float:
        """
        Compute Bellman error for current Q-function.

        Args:
            q_values: Current Q-function

        Returns:
            Maximum Bellman error across all state-action pairs
        """
        # TODO: For each state-action pair:
        #       - Compute updated Q-value using Bellman update
        #       - Calculate absolute difference from current Q-value
        # TODO: Return maximum error
        max_error = 0.0

        for s in range(self.n_states):
            for a in range(self.n_actions):
                # 終止狀態的理論值是 0
                if self.env.is_terminal(s):
                    target_q = 0.0
                else:
                    target_q = self.bellman_update(s, a, q_values)

                err = abs(q_values[s, a] - target_q)
                if err > max_error:
                    max_error = err

        return max_error

        raise NotImplementedError