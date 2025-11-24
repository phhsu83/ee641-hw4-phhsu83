"""
Evaluation script for trained multi-agent models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
import os
from multi_agent_env import MultiAgentEnv
from models import AgentDQN
import argparse


class MultiAgentEvaluator:
    """
    Evaluator for analyzing trained multi-agent policies.
    """

    def __init__(self, env: MultiAgentEnv, model_A: nn.Module, model_B: nn.Module):
        """
        Initialize evaluator.

        Args:
            env: Multi-agent environment
            model_A: Trained model for Agent A
            model_B: Trained model for Agent B
        """
        self.env = env
        self.model_A = model_A
        self.model_B = model_B
        # Use CPU for small networks
        self.device = torch.device("cpu")

        # Move models to device and set to evaluation mode
        self.model_A.to(self.device)
        self.model_B.to(self.device)
        self.model_A.eval()
        self.model_B.eval()

    def _greedy_action(self, state: np.ndarray, model: nn.Module) -> Tuple[int, float]:
        """
        Greedy action selection (no exploration).
        Returns action and comm signal.
        """
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # [1, 11]
        with torch.no_grad():
            q_values, comm = model(s)
            action = int(q_values.argmax(dim=1).item())
            comm_signal = float(comm.squeeze(0).item())
        comm_signal = float(np.clip(comm_signal, 0.0, 1.0))
        
        return action, comm_signal

    def run_episode(self, render: bool = False) -> Tuple[float, bool, Dict]:
        """
        Run single evaluation episode.

        Args:
            render: Whether to render environment

        Returns:
            reward: Episode reward
            success: Whether target was reached
            info: Episode statistics
        """
        # TODO: Reset environment
        # TODO: Initialize episode tracking
        # TODO: Run episode with greedy policy
        # TODO: Track communication patterns
        # TODO: Return results and statistics
        # Reset environment
        state_A, state_B = self.env.reset()

        episode_reward = 0.0
        done = False
        success = False

        # tracking
        positions_A: List[Tuple[int, int]] = []
        positions_B: List[Tuple[int, int]] = []
        comm_A_list: List[float] = []
        comm_B_list: List[float] = []
        dist_list: List[float] = []

        steps = 0

        while not done:
            if render:
                self.env.render()

            # Greedy policy
            action_A, comm_A = self._greedy_action(state_A, self.model_A)
            action_B, comm_B = self._greedy_action(state_B, self.model_B)

            # Step env
            (next_state_A, next_state_B), reward, done = self.env.step(
                action_A, action_B, comm_A, comm_B
            )

            episode_reward += reward
            steps += 1

            # Track positions and comms AFTER step (current positions)
            pos_A = self.env.agent_positions[0]
            pos_B = self.env.agent_positions[1]
            positions_A.append(pos_A)
            positions_B.append(pos_B)
            comm_A_list.append(comm_A)
            comm_B_list.append(comm_B)

            # Distance feature is last dim of obs (index 10)
            dist_list.append(float(next_state_A[10]))

            # terminal success check
            if done:
                if pos_A == self.env.target_pos and pos_B == self.env.target_pos:
                    success = True

            state_A, state_B = next_state_A, next_state_B

        info = {
            "steps": steps,
            "positions_A": positions_A,
            "positions_B": positions_B,
            "comm_A": comm_A_list,
            "comm_B": comm_B_list,
            "distances": dist_list,
            "target_pos": self.env.target_pos,
        }

        return episode_reward, success, info

        # raise NotImplementedError

    def evaluate_performance(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate overall performance statistics.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Statistics dictionary
        """
        # TODO: Run multiple episodes
        # TODO: Compute success rate
        # TODO: Analyze path lengths
        # TODO: Measure coordination efficiency
        # TODO: Return comprehensive statistics
        rewards = []
        successes = []
        steps_success = []
        steps_fail = []

        for _ in range(num_episodes):
            ep_r, success, info = self.run_episode(render=False)
            rewards.append(ep_r)
            successes.append(int(success))
            if success:
                steps_success.append(info["steps"])
            else:
                steps_fail.append(info["steps"])

        rewards = np.array(rewards, dtype=np.float32)
        successes = np.array(successes, dtype=np.float32)

        stats = {
            "num_episodes": num_episodes,
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "min_reward": float(rewards.min()),
            "max_reward": float(rewards.max()),
            "success_rate": float(successes.mean()),
            "mean_steps_success": float(np.mean(steps_success)) if len(steps_success) > 0 else None,
            "mean_steps_fail": float(np.mean(steps_fail)) if len(steps_fail) > 0 else None,
        }
       
        return stats

        # raise NotImplementedError

    def analyze_communication(self, num_episodes: int = 20) -> Dict:
        """
        Analyze emergent communication protocols.

        Returns:
            Communication analysis results
        """
        # TODO: Track communication signals over episodes
        # TODO: Analyze signal patterns (magnitude, variance, correlation)
        # TODO: Identify communication strategies
        # TODO: Return analysis results
        all_comm_A = []
        all_comm_B = []
        all_dist = []
        all_success = []

        for _ in range(num_episodes):
            _, success, info = self.run_episode(render=False)
            all_comm_A.extend(info["comm_A"])
            all_comm_B.extend(info["comm_B"])
            all_dist.extend(info["distances"])
            all_success.append(int(success))

        comm_A_arr = np.array(all_comm_A, dtype=np.float32)
        comm_B_arr = np.array(all_comm_B, dtype=np.float32)
        dist_arr = np.array(all_dist, dtype=np.float32)

        def safe_corr(x, y):
            if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        analysis = {
            "num_episodes": num_episodes,
            "mean_comm_A": float(comm_A_arr.mean()) if len(comm_A_arr) else 0.0,
            "mean_comm_B": float(comm_B_arr.mean()) if len(comm_B_arr) else 0.0,
            "var_comm_A": float(comm_A_arr.var()) if len(comm_A_arr) else 0.0,
            "var_comm_B": float(comm_B_arr.var()) if len(comm_B_arr) else 0.0,
            "corr_commA_dist": safe_corr(comm_A_arr, dist_arr),
            "corr_commB_dist": safe_corr(comm_B_arr, dist_arr),
            "success_rate": float(np.mean(all_success)) if len(all_success) else 0.0,
            "comm_A_hist": np.histogram(comm_A_arr, bins=10, range=(0, 1))[0].tolist(),
            "comm_B_hist": np.histogram(comm_B_arr, bins=10, range=(0, 1))[0].tolist(),
        }

        return analysis

        # raise NotImplementedError

    def visualize_trajectory(self, save_path: str = 'results/trajectory.png') -> None:
        """
        Visualize agent trajectories in an episode.

        Args:
            save_path: Path to save visualization
        """
        # TODO: Run episode while tracking positions
        # TODO: Create grid visualization
        # TODO: Plot agent paths
        # TODO: Mark key events (near target, coordination points)
        # TODO: Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        _, success, info = self.run_episode(render=False)
        grid = self.env.grid
        H, W = self.env.grid_size

        positions_A = info["positions_A"]
        positions_B = info["positions_B"]
        target_pos = info["target_pos"]

        plt.figure()
        plt.imshow(grid, interpolation="nearest")
        plt.title(f"Trajectories (success={success})")

        # paths
        if len(positions_A) > 0:
            xs_A = [p[1] for p in positions_A]
            ys_A = [p[0] for p in positions_A]
            plt.plot(xs_A, ys_A, marker="o", linewidth=2, label="Agent A")

        if len(positions_B) > 0:
            xs_B = [p[1] for p in positions_B]
            ys_B = [p[0] for p in positions_B]
            plt.plot(xs_B, ys_B, marker="s", linewidth=2, label="Agent B")

        # target marker
        plt.scatter([target_pos[1]], [target_pos[0]], marker="*", s=200, label="Target")

        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig(save_path, dpi=200)
        plt.close()

        # raise NotImplementedError

    def plot_communication_heatmap(self, save_path: str = 'results/comm_heatmap.png') -> None:
        """
        Create heatmap of communication signals across grid positions.

        Args:
            save_path: Path to save figure
        """
        # TODO: Sample communication signals at each grid position
        # TODO: Create heatmaps for both agents
        # TODO: Show correlation with distance to target
        # TODO: Save visualization
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        H, W = self.env.grid_size

        comm_sum_A = np.zeros((H, W), dtype=np.float32)
        comm_sum_B = np.zeros((H, W), dtype=np.float32)
        count_A = np.zeros((H, W), dtype=np.int32)
        count_B = np.zeros((H, W), dtype=np.int32)

        # Run many episodes to collect comm values by position
        num_collect_eps = 200
        for _ in range(num_collect_eps):
            _, _, info = self.run_episode(render=False)
            for pos, c in zip(info["positions_A"], info["comm_A"]):
                r, col = pos
                comm_sum_A[r, col] += c
                count_A[r, col] += 1
            for pos, c in zip(info["positions_B"], info["comm_B"]):
                r, col = pos
                comm_sum_B[r, col] += c
                count_B[r, col] += 1

        mean_comm_A = np.divide(comm_sum_A, np.maximum(count_A, 1))
        mean_comm_B = np.divide(comm_sum_B, np.maximum(count_B, 1))

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.title("Mean Comm Signal - Agent A")
        plt.imshow(mean_comm_A, interpolation="nearest")
        plt.gca().invert_yaxis()
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("Mean Comm Signal - Agent B")
        plt.imshow(mean_comm_B, interpolation="nearest")
        plt.gca().invert_yaxis()
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        # raise NotImplementedError

    def test_generalization(self, num_configs: int = 10) -> Dict:
        """
        Test generalization to new environment configurations.

        Args:
            num_configs: Number of test configurations

        Returns:
            Generalization performance statistics
        """
        # TODO: Generate new obstacle configurations
        # TODO: Test performance on each configuration
        # TODO: Compare to training performance
        # TODO: Return generalization metrics
        train_like_stats = self.evaluate_performance(num_episodes=100)

        per_config = []
        for k in range(num_configs):
            # New env with different seed -> different obstacles/target
            new_env = MultiAgentEnv(
                grid_size=self.env.grid_size,
                obs_window=self.env.obs_window,
                max_steps=self.env.max_steps,
                seed=1000 + k  # different from training seed
            )
            tmp_eval = MultiAgentEvaluator(new_env, self.model_A, self.model_B)
            stats_k = tmp_eval.evaluate_performance(num_episodes=50)
            per_config.append(stats_k)

        test_success_rates = [c["success_rate"] for c in per_config]
        test_mean_rewards = [c["mean_reward"] for c in per_config]

        gen_stats = {
            "num_configs": num_configs,
            "train_success_rate": train_like_stats["success_rate"],
            "train_mean_reward": train_like_stats["mean_reward"],
            "test_success_rate_mean": float(np.mean(test_success_rates)),
            "test_success_rate_std": float(np.std(test_success_rates)),
            "test_mean_reward_mean": float(np.mean(test_mean_rewards)),
            "test_mean_reward_std": float(np.std(test_mean_rewards)),
            "per_config": per_config,
        }
        
        return gen_stats

        # raise NotImplementedError


def load_trained_models(checkpoint_dir: str) -> Tuple[nn.Module, nn.Module]:
    """
    Load trained agent models from checkpoint.

    Args:
        checkpoint_dir: Directory containing saved models

    Returns:
        model_A: Agent A's trained model
        model_B: Agent B's trained model
    """
    # TODO: Load model architectures
    # TODO: Load trained weights
    # TODO: Return initialized models
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"Cannot find final checkpoints in {checkpoint_dir}. "
            f"Expected files: best_checkpoint.pth"
        )
    best_checkpoint = torch.load(checkpoint_dir, map_location="cpu")
    args = best_checkpoint.get("args", {})
    hidden_dim = args.get("hidden_dim", 64)
    input_dim = 11
    num_actions = 5
    mode = args.get("mode", "full")

    model_A = AgentDQN(input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions)
    model_B = AgentDQN(input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions)

    model_A.load_state_dict(best_checkpoint["agent_A"])
    model_B.load_state_dict(best_checkpoint["agent_B"])

    return model_A, model_B, mode

    # raise NotImplementedError


def create_evaluation_report(results: Dict, save_path: str = 'results/evaluation_report.json') -> None:
    """
    Create comprehensive evaluation report.

    Args:
        results: Evaluation results
        save_path: Path to save report
    """
    # TODO: Format results
    # TODO: Add summary statistics
    # TODO: Save as JSON report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Add a small top-level summary
    summary = {
        "performance_success_rate": results["performance"]["success_rate"],
        "performance_mean_reward": results["performance"]["mean_reward"],
        "generalization_test_success_rate_mean": results["generalization"]["test_success_rate_mean"],
    }
    results_with_summary = dict(results)
    results_with_summary["summary"] = summary

    with open(save_path, "w") as f:
        json.dump(results_with_summary, f, indent=2)

    # raise NotImplementedError


def main():
    """
    Run full evaluation suite on trained models.
    """
    # TODO: Load trained models
    # TODO: Create environment
    # TODO: Initialize evaluator
    # TODO: Run performance evaluation
    # TODO: Analyze communication
    # TODO: Test generalization
    # TODO: Create visualizations
    # TODO: Generate report
    results_dir = "results"
    evaluation_results_dir = os.path.join(results_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(evaluation_results_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description='Evaluation Multi-Agent DQN')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to saved models')
    parser.add_argument('--num_episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--render', type=bool, default=False,
                       help='Visualize episodes')

    args = parser.parse_args()
    
    # Paths
    results_dir = "results"
    checkpoint_dir = args.checkpoint

    # Load trained models
    model_A, model_B, mode = load_trained_models(checkpoint_dir)

    # Create environment (use a fresh seed to evaluate)
    env = MultiAgentEnv(grid_size=(10, 10), obs_window=3, max_steps=50, seed=999)

    # Initialize evaluator
    evaluator = MultiAgentEvaluator(env, model_A, model_B)

    # Run performance evaluation
    performance = evaluator.evaluate_performance(num_episodes=100)
    print("Performance:", performance)

    # Analyze communication
    communication = evaluator.analyze_communication(num_episodes=20)
    print("Communication analysis:", communication)

    # Test generalization
    generalization = evaluator.test_generalization(num_configs=10)
    print("Generalization:", {
        "test_success_rate_mean": generalization["test_success_rate_mean"],
        "test_success_rate_std": generalization["test_success_rate_std"],
    })

    # Create visualizations
    evaluator.visualize_trajectory(save_path=os.path.join(evaluation_results_dir, f"trajectory_{mode}.png"))
    evaluator.plot_communication_heatmap(save_path=os.path.join(evaluation_results_dir, f"communication_analysis_{mode}.png"))

    # Generate report
    results = {
        "performance": performance,
        "communication": communication,
        "generalization": generalization,
    }
    create_evaluation_report(results, save_path=os.path.join(evaluation_results_dir, f"performance_{mode}.json"))

    print(f"Saved evaluation artifacts to {evaluation_results_dir}/")

    # raise NotImplementedError


if __name__ == '__main__':
    main()