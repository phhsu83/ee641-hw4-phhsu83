"""
Experience replay buffer for multi-agent DQN training.
"""

import numpy as np
import random
from typing import Tuple, List, Optional
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores joint experiences from both agents for coordinated learning.
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, state_A: np.ndarray, state_B: np.ndarray,
             action_A: int, action_B: int,
             comm_A: float, comm_B: float,
             reward: float,
             next_state_A: np.ndarray, next_state_B: np.ndarray,
             done: bool) -> None:
        """
        Store a transition in the buffer.

        Args:
            state_A: Agent A's observation
            state_B: Agent B's observation
            action_A: Agent A's action
            action_B: Agent B's action
            comm_A: Communication from A to B
            comm_B: Communication from B to A
            reward: Shared reward
            next_state_A: Agent A's next observation
            next_state_B: Agent B's next observation
            done: Whether episode terminated
        """
        # TODO: Create transition tuple
        # TODO: Add to buffer (automatic removal of oldest if at capacity)
        # Create transition tuple
        transition = (
            state_A, state_B,
            action_A, action_B,
            comm_A, comm_B,
            reward,
            next_state_A, next_state_B,
            done
        )
        # Add to buffer (deque will automatically drop oldest if at capacity)
        self.buffer.append(transition)

        # raise NotImplementedError

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of transitions as separate arrays for each component
        """
        # TODO: Sample batch_size transitions randomly
        # TODO: Separate components into individual arrays
        # TODO: Convert to appropriate numpy arrays
        # TODO: Return tuple of arrays
        # Sample batch_size transitions randomly
        batch = random.sample(self.buffer, batch_size)

        # Unzip components
        (states_A, states_B,
         actions_A, actions_B,
         comm_As, comm_Bs,
         rewards,
         next_states_A, next_states_B,
         dones) = zip(*batch)

        # Convert to numpy arrays
        states_A = np.stack(states_A)                  # [B, obs_dim]
        states_B = np.stack(states_B)                  # [B, obs_dim]
        next_states_A = np.stack(next_states_A)
        next_states_B = np.stack(next_states_B)

        actions_A = np.array(actions_A, dtype=np.int64)
        actions_B = np.array(actions_B, dtype=np.int64)
        comm_As = np.array(comm_As, dtype=np.float32)
        comm_Bs = np.array(comm_Bs, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)      # 0.0 or 1.0

        return (states_A, states_B,
                actions_A, actions_B,
                comm_As, comm_Bs,
                rewards,
                next_states_A, next_states_B,
                dones)

        # raise NotImplementedError

    def __len__(self) -> int:
        """
        Get current size of buffer.

        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay for importance sampling.

    Samples transitions based on TD-error magnitude.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_steps: int = 100000,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_steps: Steps to anneal beta to 1.0
            seed: Random seed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.frame = 1

        # TODO: Initialize data storage
        # TODO: Initialize priority tree (sum-tree or similar)
        # TODO: Set random seed if provided
        # Data storage
        self.buffer: List = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0         # current insert position
        self.size = 0        # current number of valid transitions

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # raise NotImplementedError

    def push(self, *args, **kwargs) -> None:
        """
        Store transition with maximum priority.

        New transitions get maximum priority to ensure they're sampled at least once.
        """
        # TODO: Store transition
        # TODO: Assign maximum priority to new transition
        # args 就是一個完整的 transition，各元素順序跟 ReplayBuffer 一樣
        transition = args  # 或 tuple(args)

        # Put transition into circular buffer
        self.buffer[self.pos] = transition

        # Assign max priority to new experience
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.pos] = max_prio

        # Update position and size
        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

        # raise NotImplementedError

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch with prioritization.

        Returns:
            transitions: Batch of transitions
            weights: Importance sampling weights
            indices: Indices for updating priorities
        """
        # TODO: Update beta based on schedule
        # TODO: Sample transitions based on priorities
        # TODO: Calculate importance sampling weights
        # TODO: Return transitions, weights, and indices
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        # Update beta based on schedule
        # beta anneals from beta_start to 1.0 over beta_steps calls
        self.beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.frame / float(self.beta_steps)
        )
        self.frame += 1

        # Compute probabilities from priorities^alpha
        prios = self.priorities[:self.size]
        if prios.sum() == 0:
            # Avoid division by zero; fall back to uniform
            probs = np.ones_like(prios) / len(prios)
        else:
            scaled_prios = prios ** self.alpha
            probs = scaled_prios / scaled_prios.sum()

        # Sample indices according to probabilities
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Gather transitions
        batch = [self.buffer[idx] for idx in indices]

        (states_A, states_B,
         actions_A, actions_B,
         comm_As, comm_Bs,
         rewards,
         next_states_A, next_states_B,
         dones) = zip(*batch)

        # Convert to numpy arrays
        states_A = np.stack(states_A)
        states_B = np.stack(states_B)
        next_states_A = np.stack(next_states_A)
        next_states_B = np.stack(next_states_B)

        actions_A = np.array(actions_A, dtype=np.int64)
        actions_B = np.array(actions_B, dtype=np.int64)
        comm_As = np.array(comm_As, dtype=np.float32)
        comm_Bs = np.array(comm_Bs, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Importance sampling weights
        # w_i = (1 / (N * P(i)))^beta, normalized by max weight
        N = self.size
        sample_probs = probs[indices]
        weights = (N * sample_probs) ** (-self.beta)
        weights /= weights.max()  # normalize to 1.0 max
        weights = weights.astype(np.float32)

        transitions = (states_A, states_B,
                       actions_A, actions_B,
                       comm_As, comm_Bs,
                       rewards,
                       next_states_A, next_states_B,
                       dones)

        return transitions, weights, indices

        # raise NotImplementedError

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically TD-errors)
        """
        # TODO: Update priorities for given indices
        # TODO: Apply alpha exponent for prioritization
        # Small epsilon to avoid zero priority
        eps = 1e-6
        priorities = np.abs(priorities) + eps

        for idx, prio in zip(indices, priorities):
            if 0 <= idx < self.size:
                self.priorities[idx] = float(prio)

        # raise NotImplementedError