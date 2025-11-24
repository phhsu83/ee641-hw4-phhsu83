"""
Training script for multi-agent DQN with communication.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
from typing import Tuple, Optional
from multi_agent_env import MultiAgentEnv
from models import AgentDQN
from replay_buffer import ReplayBuffer
import random


def apply_observation_mask(obs: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply masking to observation based on ablation mode.

    Args:
        obs: 11-dimensional observation vector
        mode: One of 'independent', 'comm', 'full'

    Returns:
        Masked observation
    """
    # TODO: Implement masking logic
    # 'independent': Set elements 9 and 10 to zero
    # 'comm': Set element 10 to zero
    # 'full': No masking
    masked = obs.copy()
    # indices:
    # 0-8: 3x3 patch
    # 9:   comm from other agent
    # 10:  normalized distance
    if mode == 'independent':
        masked[9] = 0.0
        masked[10] = 0.0
    elif mode == 'comm':
        masked[10] = 0.0
    elif mode == 'full':
        pass  # no masking
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return masked

    # raise NotImplementedError


class MultiAgentTrainer:
    """
    Trainer for multi-agent DQN system.

    Handles training loop, exploration, and network updates.
    """

    def __init__(self, env: MultiAgentEnv, args):
        """
        Initialize trainer.

        Args:
            env: Multi-agent environment
            args: Training arguments
        """
        self.env = env
        self.args = args

        # Use CPU for small networks
        self.device = torch.device("cpu")

        # TODO: Initialize networks for both agents (remember to .to(self.device))
        # TODO: Initialize target networks (if using)
        # TODO: Initialize optimizers
        # TODO: Initialize replay buffer
        # TODO: Initialize epsilon for exploration
        # Networks for both agents
        # Observation dim is 11 (3x3 patch + comm + distance)
        self.network_A = AgentDQN(
            input_dim=11,
            hidden_dim=args.hidden_dim,
            num_actions=5
        ).to(self.device)

        self.network_B = AgentDQN(
            input_dim=11,
            hidden_dim=args.hidden_dim,
            num_actions=5
        ).to(self.device)

        # Target networks
        self.target_A = AgentDQN(
            input_dim=11,
            hidden_dim=args.hidden_dim,
            num_actions=5
        ).to(self.device)

        self.target_B = AgentDQN(
            input_dim=11,
            hidden_dim=args.hidden_dim,
            num_actions=5
        ).to(self.device)

        # Initialize target networks
        self.target_A.load_state_dict(self.network_A.state_dict())
        self.target_B.load_state_dict(self.network_B.state_dict())

        # Optimizers
        self.optimizer_A = optim.Adam(self.network_A.parameters(), lr=args.lr)
        self.optimizer_B = optim.Adam(self.network_B.parameters(), lr=args.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000, seed=args.seed)

        # Epsilon for exploration
        self.epsilon = args.epsilon_start

        # For logging
        self.global_step = 0

        # raise NotImplementedError

    def select_action(self, state: np.ndarray, network: nn.Module,
                      epsilon: float) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Agent observation (11-dimensional, may need masking)
            network: Agent's DQN
            epsilon: Exploration probability

        Returns:
            action: Selected action
            comm_signal: Communication signal
        """
        # TODO: Apply observation masking based on self.args.mode
        #       masked_state = apply_observation_mask(state, self.args.mode)
        # TODO: With probability epsilon, select random action
        # TODO: Otherwise, select action with highest Q-value
        # TODO: Always get communication signal from network
        # TODO: Return (action, comm_signal)
        # Apply observation masking based on experiment mode
        masked_state = apply_observation_mask(state, self.args.mode)

        # With probability epsilon, random action
        num_actions = 5
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
            # 仍然從網路算一次 comm_signal（讓通訊行為跟 state 一致）
            with torch.no_grad():
                s = torch.from_numpy(masked_state).float().unsqueeze(0).to(self.device)
                _, comm = network(s)
                comm_signal = comm.squeeze(0).item()
        else:
            # Greedy action from Q-network
            with torch.no_grad():
                s = torch.from_numpy(masked_state).float().unsqueeze(0).to(self.device)
                q_values, comm = network(s)
                action = q_values.argmax(dim=1).item()
                comm_signal = comm.squeeze(0).item()

        # Clamp comm_signal to [0,1] just in case
        comm_signal = float(np.clip(comm_signal, 0.0, 1.0))
        
        return action, comm_signal

        # raise NotImplementedError

    def _mask_batch_states(self, states_A, states_B, next_states_A, next_states_B):
        """
        Apply the same masking as in interaction, but on batched observations.
        states_* are numpy arrays of shape [B, 11].
        """
        mode = self.args.mode

        if mode == 'independent':
            states_A[:, 9:] = 0.0
            states_B[:, 9:] = 0.0
            next_states_A[:, 9:] = 0.0
            next_states_B[:, 9:] = 0.0
        elif mode == 'comm':
            states_A[:, 10] = 0.0
            states_B[:, 10] = 0.0
            next_states_A[:, 10] = 0.0
            next_states_B[:, 10] = 0.0
        elif mode == 'full':
            pass

    def update_networks(self, batch_size: int) -> float:
        """
        Sample batch and update both agent networks.

        Args:
            batch_size: Size of training batch

        Returns:
            loss: Combined loss value
        """
        # TODO: Sample batch from replay buffer
        # TODO: Convert to tensors and move to device
        # TODO: Compute Q-values for current states
        # TODO: Compute target Q-values using target networks
        # TODO: Calculate TD loss for both agents
        # TODO: Backpropagate and update networks
        # TODO: Return combined loss
        if len(self.replay_buffer) < batch_size:
            return 0.0

        gamma = self.args.gamma

        # Sample batch
        (states_A, states_B,
         actions_A, actions_B,
         comm_As, comm_Bs,
         rewards,
         next_states_A, next_states_B,
         dones) = self.replay_buffer.sample(batch_size)

        # Apply masking (same as at acting time)
        self._mask_batch_states(states_A, states_B, next_states_A, next_states_B)

        # Convert to tensors
        states_A = torch.from_numpy(states_A).float().to(self.device)
        states_B = torch.from_numpy(states_B).float().to(self.device)
        next_states_A = torch.from_numpy(next_states_A).float().to(self.device)
        next_states_B = torch.from_numpy(next_states_B).float().to(self.device)

        actions_A = torch.from_numpy(actions_A).long().to(self.device)
        actions_B = torch.from_numpy(actions_B).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # --- Agent A ---
        q_values_A, _ = self.network_A(states_A)           # [B, num_actions]
        q_values_A = q_values_A.gather(1, actions_A.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values_A, _ = self.target_A(next_states_A)
            max_next_q_A = next_q_values_A.max(dim=1)[0]
            target_A = rewards + gamma * (1.0 - dones) * max_next_q_A

        loss_A = F.mse_loss(q_values_A, target_A)

        # --- Agent B ---
        q_values_B, _ = self.network_B(states_B)
        q_values_B = q_values_B.gather(1, actions_B.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values_B, _ = self.target_B(next_states_B)
            max_next_q_B = next_q_values_B.max(dim=1)[0]
            target_B = rewards + gamma * (1.0 - dones) * max_next_q_B

        loss_B = F.mse_loss(q_values_B, target_B)

        total_loss = loss_A + loss_B

        # Backprop and update networks
        self.optimizer_A.zero_grad()
        self.optimizer_B.zero_grad()
        total_loss.backward()
        self.optimizer_A.step()
        self.optimizer_B.step()

        return float(total_loss.item())

        # raise NotImplementedError

    def train_episode(self) -> Tuple[float, bool]:
        """
        Run one training episode.

        Returns:
            episode_reward: Total reward for episode
            success: Whether agents reached target
        """
        # TODO: Reset environment
        # TODO: Initialize episode variables
        # TODO: Run episode until termination:
        #       - Select actions for both agents
        #       - Execute actions in environment
        #       - Store transition in replay buffer
        #       - Update networks if enough samples
        # TODO: Return episode reward and success flag
        self.network_A.train()
        self.network_B.train()

        state_A, state_B = self.env.reset()
        episode_reward = 0.0
        success = False

        done = False
        while not done:
            # Select actions for both agents
            action_A, comm_A = self.select_action(state_A, self.network_A, self.epsilon)
            action_B, comm_B = self.select_action(state_B, self.network_B, self.epsilon)

            # Step environment
            (next_state_A, next_state_B), reward, done = self.env.step(
                action_A, action_B, comm_A, comm_B
            )

            episode_reward += reward

            # Store transition in replay buffer (store raw obs, masking在訓練時做)
            self.replay_buffer.push(
                state_A, state_B,
                action_A, action_B,
                comm_A, comm_B,
                reward,
                next_state_A, next_state_B,
                done
            )

            # Update networks if enough samples
            if len(self.replay_buffer) >= self.args.batch_size:
                self.update_networks(self.args.batch_size)
            
            self.global_step += 1

            # Move to next state
            state_A, state_B = next_state_A, next_state_B

            # 判斷有沒有成功到達 target（簡單用 reward 判斷：同時到 target 會給 +10）
            #if done and reward > 5.0:
            #    success = True
            if done:
                pos_A = self.env.agent_positions[0]
                pos_B = self.env.agent_positions[1]
                if pos_A == self.env.target_pos and pos_B == self.env.target_pos:
                    success = True

        return episode_reward, success

        # raise NotImplementedError

    def train(self) -> None:
        """
        Main training loop.
        """
        # TODO: Create results directories
        # TODO: Initialize logging
        # TODO: Main training loop:
        #       - Run episodes
        #       - Update epsilon
        #       - Update target networks periodically
        #       - Log progress
        #       - Save checkpoints
        # TODO: Save final models including TorchScript format:
        #       scripted_model = torch.jit.script(self.network_A)
        #       scripted_model.save("dqn_net.pt")
        results_dir = "results"
        training_logs_dir = os.path.join(results_dir, "training_log")
        agent_models_dir = os.path.join(results_dir, "agent_models")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(training_logs_dir, exist_ok=True)
        os.makedirs(agent_models_dir, exist_ok=True)

        log = {
            "episode_rewards": [],
            "success_flags": [],
            "epsilon": [],
        }

        for ep in range(1, self.args.num_episodes + 1):
            ep_reward, success = self.train_episode()

            # Epsilon decay
            
            self.epsilon = max(
                self.args.epsilon_end,
                self.epsilon * self.args.epsilon_decay
            )

            log["episode_rewards"].append(ep_reward)
            log["success_flags"].append(int(success))
            log["epsilon"].append(self.epsilon)

            # Update target networks periodically
            if ep % self.args.target_update == 0:
                self.target_A.load_state_dict(self.network_A.state_dict())
                self.target_B.load_state_dict(self.network_B.state_dict())

            # Logging
            if ep % 50 == 0:
                recent_rewards = log["episode_rewards"][-50:]
                recent_success = log["success_flags"][-50:]
                avg_r = np.mean(recent_rewards)
                succ_rate = np.mean(recent_success)
                print(f"Episode {ep}/{self.args.num_episodes} | "
                      f"AvgReward(50ep)={avg_r:.2f} | "
                      f"Success(50ep)={succ_rate:.2f} | "
                      f"Epsilon={self.epsilon:.3f}")

            # Save checkpoints
            if ep % self.args.save_freq == 0:
                torch.save(self.network_A.state_dict(),
                           os.path.join(agent_models_dir, f"agentA_ep{ep}_{self.args.mode}.pt"))
                torch.save(self.network_B.state_dict(),
                           os.path.join(agent_models_dir, f"agentB_ep{ep}_{self.args.mode}.pt"))

        # Save log so far
        with open(os.path.join(training_logs_dir, f"train_log_{self.args.mode}.json"), "w") as f:
            json.dump(log, f)

        # Save final models and TorchScript
        '''
        torch.save(self.network_A.state_dict(),
                   os.path.join(agent_models_dir, f"agentA_final_{self.args.mode}.pt"))
        torch.save(self.network_B.state_dict(),
                   os.path.join(agent_models_dir, f"agentB_final_{self.args.mode}.pt"))
        '''
        best_checkpoint = {
            "agent_A": self.network_A.state_dict(),
            "agent_B": self.network_B.state_dict(),
            "args": vars(self.args),
        }
        torch.save(best_checkpoint,os.path.join(agent_models_dir, f'best_checkpoint_{self.args.mode}.pth'))

        # TorchScript export (只示範 agent A)
        scripted_model_A = torch.jit.script(self.network_A.cpu())
        scripted_model_A.save(os.path.join(agent_models_dir, f"dqn_agentA_scripted_{self.args.mode}.pt"))
        scripted_model_B = torch.jit.script(self.network_B.cpu())
        scripted_model_B.save(os.path.join(agent_models_dir, f"dqn_agentB_scripted_{self.args.mode}.pt"))

        # raise NotImplementedError

    def evaluate(self, num_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            mean_reward: Average reward
            success_rate: Fraction of successful episodes
        """
        # TODO: Set networks to evaluation mode
        # TODO: Run episodes without exploration
        # TODO: Track rewards and successes
        # TODO: Return statistics
        self.network_A.eval()
        self.network_B.eval()

        rewards = []
        successes = []

        for _ in range(num_episodes):
            state_A, state_B = self.env.reset()
            ep_reward = 0.0
            done = False
            success = False

            while not done:
                # evaluation 無探索：epsilon=0
                action_A, comm_A = self.select_action(state_A, self.network_A, epsilon=0.0)
                action_B, comm_B = self.select_action(state_B, self.network_B, epsilon=0.0)

                (next_state_A, next_state_B), reward, done = self.env.step(
                    action_A, action_B, comm_A, comm_B
                )
                ep_reward += reward
                state_A, state_B = next_state_A, next_state_B

                if done:
                    pos_A = self.env.agent_positions[0]
                    pos_B = self.env.agent_positions[1]
                    if pos_A == self.env.target_pos and pos_B == self.env.target_pos:
                        success = True

            rewards.append(ep_reward)
            successes.append(int(success))

        mean_reward = float(np.mean(rewards))
        success_rate = float(np.mean(successes))
        
        return mean_reward, success_rate

        # raise NotImplementedError


def main():
    """
    Parse arguments and run training.
    """
    parser = argparse.ArgumentParser(description='Train Multi-Agent DQN')

    # Environment parameters
    parser.add_argument('--grid_size', type=int, nargs=2, default=[10, 10],
                       help='Grid dimensions')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode')

    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')

    # Exploration parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                       help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='Epsilon decay rate')

    # Network parameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer size')
    parser.add_argument('--target_update', type=int, default=100,
                       help='Target network update frequency')

    # Ablation study mode
    parser.add_argument('--mode', type=str, default='full',
                       choices=['independent', 'comm', 'full'],
                       help='Information mode: independent (mask comm+dist), '
                            'comm (mask dist only), full (no masking)')

    # Other parameters
    parser.add_argument('--seed', type=int, default=641,
                       help='Random seed')
    parser.add_argument('--save_freq', type=int, default=500,
                       help='Model save frequency')

    args = parser.parse_args()

    # TODO: Set random seeds
    # TODO: Create environment
    # TODO: Create trainer
    # TODO: Run training
    # TODO: Final evaluation
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env = MultiAgentEnv(
        grid_size=tuple(args.grid_size),
        obs_window=3,
        max_steps=args.max_steps,
        seed=args.seed
    )

    # Create trainer
    trainer = MultiAgentTrainer(env, args)

    # Run training
    trainer.train()

    # Final evaluation
    mean_reward, success_rate = trainer.evaluate(num_episodes=20)
    
    print(f"Final evaluation - mean reward: {mean_reward:.2f}, "
          f"success rate: {success_rate:.2f}")

    # raise NotImplementedError


if __name__ == '__main__':
    main()