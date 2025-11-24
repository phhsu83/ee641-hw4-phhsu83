"""
Visualization utilities for gridworld and policies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Optional, Tuple
import os


class GridWorldVisualizer:
    """
    Visualizer for gridworld environment, value functions, and policies.
    """

    def __init__(self, grid_size: int = 5):
        """
        Initialize visualizer.

        Args:
            grid_size: Size of grid
        """
        self.grid_size = grid_size

        # Define special positions
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]

    def _mark_special_cells(self, ax):
        # start
        r, c = self.start_pos
        ax.text(c, r - 0.2, "S", ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")

        # goal
        r, c = self.goal_pos
        ax.text(c, r - 0.2, "G", ha="center", va="center", color="yellow",
                fontsize=10, fontweight="bold")

        # obstacles
        for (r, c) in self.obstacles:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                       facecolor="black", alpha=0.5))
            ax.text(c, r - 0.2, "#", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")

        # penalties
        for (r, c) in self.penalties:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                       facecolor="red", alpha=0.3))
            ax.text(c, r - 0.2, "P", ha="center", va="center", color="red",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)  # y 軸反轉，row 0 在上方
        ax.set_aspect("equal")


    def plot_value_function(self, values: np.ndarray, title: str = "Value Function") -> None:
        """
        Plot value function as heatmap.

        Args:
            values: Value function V(s) for each state
            title: Plot title
        """
        # TODO: Reshape values to 2D grid
        # TODO: Create heatmap with appropriate colormap
        # TODO: Mark special cells (start, goal, obstacles, penalties)
        # TODO: Add colorbar and labels
        # TODO: Save figure to results/visualizations/
        values = np.asarray(values)
        grid = values.reshape(self.grid_size, self.grid_size)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(grid, cmap="viridis")

        self._mark_special_cells(ax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)")

        # 每格數值文字（可選）
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                ax.text(c, r, f"{grid[r, c]:.2f}",
                        ha="center", va="center", color="white", fontsize=7)

        fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.tight_layout()
        fig.savefig(f"results/visualizations/{fname}.png", dpi=150)
        plt.close(fig)

        # raise NotImplementedError

    def plot_policy(self, policy: np.ndarray, title: str = "Optimal Policy") -> None:
        """
        Plot policy with arrows showing optimal actions.

        Args:
            policy: Array of optimal actions for each state
            title: Plot title
        """
        # TODO: Create grid plot
        # TODO: For each state:
        #       - Draw arrow indicating action direction
        #       - Handle special cells appropriately
        # TODO: Mark start, goal, obstacles, penalties
        # TODO: Save figure to results/visualizations/
        policy = np.asarray(policy).reshape(-1)

        fig, ax = plt.subplots(figsize=(5, 5))

        # 背景淡灰色格線
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                           edgecolor="lightgray", facecolor="none"))

        # 動作對應箭頭
        action_to_arrow = {
            0: "↑",  # UP
            1: "→",  # RIGHT
            2: "↓",  # DOWN
            3: "←",  # LEFT
        }

        for s in range(self.grid_size * self.grid_size):
            r = s // self.grid_size
            c = s % self.grid_size
            pos = (r, c)

            # 特殊格子不要畫 policy 箭頭（改用文字）
            if pos == self.goal_pos or pos in self.obstacles:
                continue

            a = int(policy[s])
            arrow = action_to_arrow.get(a, ".")

            ax.text(c, r, arrow, ha="center", va="center",
                    fontsize=12, color="black")

        self._mark_special_cells(ax)
        ax.set_title(title)

        fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.tight_layout()
        fig.savefig(f"results/visualizations/{fname}.png", dpi=150)
        plt.close(fig)

        # raise NotImplementedError

    def plot_q_function(self, q_values: np.ndarray, title: str = "Q-Function") -> None:
        """
        Plot Q-function with multiple subplots for each action.

        Args:
            q_values: Q-function Q(s,a)
            title: Plot title
        """
        # TODO: Create subplot for each action
        # TODO: For each action:
        #       - Show Q-values as heatmap
        #       - Mark special cells
        # TODO: Add overall title and save
        q_values = np.asarray(q_values)
        assert q_values.shape[1] == 4, "Expected 4 actions (UP, RIGHT, DOWN, LEFT)"

        action_names = ["UP", "RIGHT", "DOWN", "LEFT"]

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
        axes = axes.flatten()

        for a in range(4):
            grid = q_values[:, a].reshape(self.grid_size, self.grid_size)
            ax = axes[a]
            im = ax.imshow(grid, cmap="viridis")
            self._mark_special_cells(ax)
            ax.set_title(f"Q(s,a) - {action_names[a]}")

            # 每格數值文字（可選）
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    ax.text(c, r, f"{grid[r, c]:.2f}",
                            ha="center", va="center", color="white", fontsize=6)

        fig.suptitle(title)
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)

        fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(f"results/visualizations/{fname}.png", dpi=150)
        plt.close(fig)

        # raise NotImplementedError

    def plot_convergence(self, vi_history: list, qi_history: list) -> None:
        """
        Plot convergence curves for both algorithms.

        Args:
            vi_history: Value iteration convergence history
            qi_history: Q-iteration convergence history
        """
        # TODO: Plot Bellman error vs iteration for both algorithms
        # TODO: Use log scale for y-axis
        # TODO: Add legend and labels
        # TODO: Save figure
        if not vi_history and not qi_history:
            print("No convergence history provided, skipping convergence plot.")
            return

        fig, ax = plt.subplots(figsize=(6, 4))

        if vi_history:
            ax.plot(range(1, len(vi_history) + 1),
                    vi_history, label="Value Iteration")
        if qi_history:
            ax.plot(range(1, len(qi_history) + 1),
                    qi_history, label="Q-Iteration")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Bellman Error")
        ax.set_yscale("log")
        ax.set_title("Convergence of Bellman Error")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        fig.tight_layout()
        fig.savefig("results/visualizations/convergence.png", dpi=150)
        plt.close(fig)
        
        # raise NotImplementedError

    def create_comparison_figure(self, vi_values: np.ndarray, qi_values: np.ndarray,
                                vi_policy: np.ndarray, qi_policy: np.ndarray) -> None:
        """
        Create comparison figure showing both algorithms' results.

        Args:
            vi_values: Value function from Value Iteration
            qi_values: Value function from Q-Iteration
            vi_policy: Policy from Value Iteration
            qi_policy: Policy from Q-Iteration
        """
        # TODO: Create 2x2 subplot
        #       - Top left: VI value function
        #       - Top right: QI value function
        #       - Bottom left: VI policy
        #       - Bottom right: QI policy
        # TODO: Highlight any differences
        # TODO: Save comprehensive comparison figure
        vi_values = np.asarray(vi_values).reshape(self.grid_size, self.grid_size)
        qi_values = np.asarray(qi_values).reshape(self.grid_size, self.grid_size)
        vi_policy = np.asarray(vi_policy).reshape(-1)
        qi_policy = np.asarray(qi_policy).reshape(-1)

        action_to_arrow = {
            0: "↑",
            1: "→",
            2: "↓",
            3: "←",
        }

        fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

        # Top-left: VI values
        ax = axes[0, 0]
        im1 = ax.imshow(vi_values, cmap="viridis")
        self._mark_special_cells(ax)
        ax.set_title("VI Value Function")
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                ax.text(c, r, f"{vi_values[r, c]:.2f}",
                        ha="center", va="center", color="white", fontsize=6)

        # Top-right: QI values
        ax = axes[0, 1]
        im2 = ax.imshow(qi_values, cmap="viridis")
        self._mark_special_cells(ax)
        ax.set_title("QI Value Function")
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                ax.text(c, r, f"{qi_values[r, c]:.2f}",
                        ha="center", va="center", color="white", fontsize=6)

        fig.colorbar(im1, ax=[axes[0, 0], axes[0, 1]],
                     fraction=0.02, pad=0.02, label="V(s)")

        # Bottom-left: VI policy
        ax = axes[1, 0]
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                s = r * self.grid_size + c
                pos = (r, c)
                if pos == self.goal_pos or pos in self.obstacles:
                    continue
                a = int(vi_policy[s])
                arrow = action_to_arrow.get(a, ".")
                ax.text(c, r, arrow, ha="center", va="center",
                        fontsize=12, color="black")
        self._mark_special_cells(ax)
        ax.set_title("VI Policy")

        # Bottom-right: QI policy
        ax = axes[1, 1]
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                s = r * self.grid_size + c
                pos = (r, c)
                if pos == self.goal_pos or pos in self.obstacles:
                    continue
                a = int(qi_policy[s])
                arrow = action_to_arrow.get(a, ".")
                ax.text(c, r, arrow, ha="center", va="center",
                        fontsize=12, color="black")
        self._mark_special_cells(ax)
        ax.set_title("QI Policy")

        fig.suptitle("Value Iteration vs Q-Iteration Comparison", fontsize=14)
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig("results/visualizations/comparison_vi_qi.png", dpi=150)
        plt.close(fig)

        # raise NotImplementedError


def visualize_results():
    """
    Load and visualize saved results from training.
    """
    # TODO: Load saved value functions and policies
    # TODO: Create visualizer instance
    # TODO: Generate all visualization plots
    # TODO: Print summary statistics
    os.makedirs("results/visualizations", exist_ok=True)

    # 1. 載入儲存的結果
    vf_path = "results/value_function.npz"
    qf_path = "results/q_function.npz"
    pol_path = "results/optimal_policy.npz"

    if not (os.path.exists(vf_path) and os.path.exists(qf_path) and os.path.exists(pol_path)):
        print("Saved result files not found. Please run training script first.")
        return

    vf_data = np.load(vf_path)
    qf_data = np.load(qf_path)
    pol_data = np.load(pol_path)

    vi_values = vf_data["values"]
    vi_iters = int(vf_data["iterations"])

    qi_q_values = qf_data["q_values"]
    qi_values = qf_data["values"]
    qi_iters = int(qf_data["iterations"])

    vi_policy = pol_data["policy_vi"]
    qi_policy = pol_data["policy_qi"]

    grid_size = int(np.sqrt(len(vi_values)))
    visualizer = GridWorldVisualizer(grid_size=grid_size)

    # 2. 生成各種視覺化
    visualizer.plot_value_function(vi_values, title="Value Function VI")
    visualizer.plot_value_function(qi_values, title="Value Function QI")
    visualizer.plot_policy(vi_policy, title="Optimal Policy VI")
    visualizer.plot_policy(qi_policy, title="Optimal Policy QI")
    visualizer.plot_q_function(qi_q_values, title="Q Function")

    # 若之後有存收斂歷史，可載入並畫出收斂曲線（這裡先留介面）
    conv_path = "results/convergence.npz"
    if os.path.exists(conv_path):
        conv_data = np.load(conv_path)
        vi_hist = conv_data.get("vi_history", []).tolist()
        qi_hist = conv_data.get("qi_history", []).tolist()
        visualizer.plot_convergence(vi_hist, qi_hist)

    # 綜合比較圖
    visualizer.create_comparison_figure(vi_values, qi_values, vi_policy, qi_policy)

    # 3. 印 summary
    policies_match = np.array_equal(vi_policy, qi_policy)
    max_value_diff = float(np.max(np.abs(vi_values - qi_values)))

    print("=== Visualization Summary ===")
    print(f"Value Iteration iterations: {vi_iters}")
    print(f"Q-Iteration iterations:     {qi_iters}")
    print(f"Policies match:             {policies_match}")
    print(f"Max |V_vi - V_q_from_Q|:    {max_value_diff:.6f}")
    print("Figures saved to results/visualizations/")

    # raise NotImplementedError


if __name__ == '__main__':
    visualize_results()