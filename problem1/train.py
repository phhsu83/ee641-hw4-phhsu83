"""
Training script for Value Iteration and Q-Iteration.
"""

import numpy as np
import argparse
import json
import os
from environment import GridWorldEnv
from value_iteration import ValueIteration
from q_iteration import QIteration


def main():
    """
    Run both algorithms and save results.
    """
    parser = argparse.ArgumentParser(description='Train RL algorithms on GridWorld')
    parser.add_argument('--seed', type=int, default=641, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Convergence threshold')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations')
    args = parser.parse_args()

    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)

    # TODO: Initialize environment with seed
    # TODO: Run Value Iteration
    #       - Create ValueIteration solver
    #       - Solve for optimal values
    #       - Extract policy
    #       - Save results
    # TODO: Run Q-Iteration
    #       - Create QIteration solver
    #       - Solve for optimal Q-values
    #       - Extract policy and values
    #       - Save results
    # TODO: Compare algorithms
    #       - Print convergence statistics
    #       - Check if policies match
    #       - Save comparison results

    # 1. Initialize environment
    env = GridWorldEnv(seed=args.seed)

    # 2. Run Value Iteration
    vi_solver = ValueIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    vi_values, vi_iters = vi_solver.solve(max_iterations=args.max_iter)
    vi_policy = vi_solver.extract_policy(vi_values)

    # Save value function results
    np.savez(
        'results/value_function.npz',
        values=vi_values,
        iterations=vi_iters,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
        algorithm='value_iteration',
    )

    # 3. Run Q-Iteration
    qi_solver = QIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    qi_q_values, qi_iters = qi_solver.solve(max_iterations=args.max_iter)
    qi_policy = qi_solver.extract_policy(qi_q_values)
    qi_values = qi_solver.extract_values(qi_q_values)

    # Save Q-function results
    np.savez(
        'results/q_function.npz',
        q_values=qi_q_values,
        values=qi_values,
        iterations=qi_iters,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
        algorithm='q_iteration',
    )

    # 4. Save optimal policies
    np.savez(
        'results/optimal_policy.npz',
        policy_vi=vi_policy,
        policy_qi=qi_policy,
    )

    # 5. Compare algorithms
    policies_match = np.array_equal(vi_policy, qi_policy)
    max_value_diff = float(np.max(np.abs(vi_values - qi_values)))

    print("=== Comparison ===")
    print(f"Value Iteration iterations: {vi_iters}")
    print(f"Q-Iteration iterations:     {qi_iters}")
    print(f"Policies match:             {policies_match}")
    print(f"Max |V_vi - V_q_from_Q|:    {max_value_diff:.6f}")

    comparison_results = {
        "seed": args.seed,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "max_iter": args.max_iter,
        "value_iteration": {
            "iterations": int(vi_iters),
        },
        "q_iteration": {
            "iterations": int(qi_iters),
        },
        "policies_match": bool(policies_match),
        "max_value_difference": max_value_diff,
    }

    with open('results/comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)

    # raise NotImplementedError


if __name__ == '__main__':
    main()