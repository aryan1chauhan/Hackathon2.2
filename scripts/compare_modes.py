import os
import sys
import json
from statistics import mean

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.runner import run_experiment, dummy_llm_client, dummy_validator
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd


def compare(n_runs=3, steps=200, seed_base=0, urgency_weights=None):
    if urgency_weights is None:
        urgency_weights = [0.0, 1.0, 2.0, 4.0, 8.0]

    # Run baseline greedy for reference
    greedy_results = []
    for i in range(n_runs):
        seed = seed_base + i
        _, trace = run_experiment(mode='greedy', max_steps=steps, seed=seed)
        greedy_results.append(trace['summary'])
        print(f'Run greedy seed={seed}:', trace['summary'])

    # Run LLM coordinated mode
    llm_results = []
    for i in range(n_runs):
        seed = seed_base + i
        _, trace = run_experiment(mode='coordinated', max_steps=steps, seed=seed, llm_client=dummy_llm_client, validator=dummy_validator)
        llm_results.append(trace['summary'])
        print(f'Run LLM coordinated seed={seed}:', trace['summary'])

    # Summarize
    def summarize(arr):
        return {
            'avg_rescued': float(np.mean([x['total_rescued'] for x in arr])),
            'avg_died': float(np.mean([x['total_died'] for x in arr])),
            'avg_spawned': float(np.mean([x['total_spawned'] for x in arr])),
            'avg_steps': float(np.mean([x['steps'] for x in arr])),
            'avg_wait_time': float(np.mean([x.get('avg_wait_time', 0) for x in arr])),
            'avg_llm_success_rate': float(np.mean([x.get('llm_success_rate', 0) for x in arr])),
            'avg_fallback_count': float(np.mean([x.get('fallback_count', 0) for x in arr]))
        }

    greedy_summary = summarize(greedy_results)
    llm_summary = summarize(llm_results)

    # Save metrics to CSV
    all_results = []
    for res in greedy_results:
        res['mode'] = 'greedy'
        all_results.append(res)
    for res in llm_results:
        res['mode'] = 'llm_coordinated'
        all_results.append(res)

    df = pd.DataFrame(all_results)
    csv_out = os.path.join('examples', 'comparison_metrics.csv')
    df.to_csv(csv_out, index=False)
    print(f'Saved metrics to {csv_out}')

    # Calculate Jain fairness index for agent workloads
    def jain_fairness(workloads):
        if not workloads:
            return 0.0
        n = len(workloads)
        if n == 0:
            return 0.0
        sum_x = sum(workloads)
        sum_x2 = sum(x**2 for x in workloads)
        if sum_x == 0:
            return 1.0  # All zero is perfectly fair
        return (sum_x ** 2) / (n * sum_x2)

    # Compute fairness for each run
    greedy_fairness = []
    llm_fairness = []
    for res in greedy_results:
        workloads = list(res.get('agent_workloads', {}).values())
        greedy_fairness.append(jain_fairness(workloads))
    for res in llm_results:
        workloads = list(res.get('agent_workloads', {}).values())
        llm_fairness.append(jain_fairness(workloads))

    avg_fairness_greedy = np.mean(greedy_fairness) if greedy_fairness else 0.0
    avg_fairness_llm = np.mean(llm_fairness) if llm_fairness else 0.0

    # Generate Efficiency vs Fairness plot
    efficiency_greedy = greedy_summary['avg_rescued']
    efficiency_llm = llm_summary['avg_rescued']

    plt.figure(figsize=(8, 6))
    plt.scatter(avg_fairness_greedy, efficiency_greedy, color='red', marker='o', s=100, label='Greedy Baseline')
    plt.scatter(avg_fairness_llm, efficiency_llm, color='blue', marker='x', s=100, label='LLM Coordinator')

    plt.xlabel('Jain Fairness Index')
    plt.ylabel('Efficiency (Avg Rescued)')
    plt.title('Efficiency vs Fairness')
    plt.legend()
    plt.grid(True)
    plot_out = os.path.join('examples', 'efficiency_fairness_plot.png')
    plt.savefig(plot_out)

    print('\nComparison Summary:')
    print(json.dumps({'greedy': greedy_summary, 'llm_coordinated': llm_summary}, indent=2))
    print(f'Saved plot to {plot_out}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-runs', type=int, default=3, help='number of seeds per weight')
    parser.add_argument('--steps', type=int, default=200, help='steps per run')
    parser.add_argument('--seed-base', type=int, default=0, help='starting seed')
    parser.add_argument('--weights', type=float, nargs='*', default=[],
                        help='urgency weight values to sweep (not used for LLM comparison)')

    args = parser.parse_args()
    compare(n_runs=args.n_runs, steps=args.steps, seed_base=args.seed_base)
