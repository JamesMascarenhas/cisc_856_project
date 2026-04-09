# RL Algorithms Comparison on FrozenLake

Comparison of planning and reinforcement learning algorithms on the FrozenLake-v1 environment (Gymnasium). All algorithms use the same pre-generated maps for fair comparison.

## Algorithms

### DQN — Deep Q-Network
SB3-based DQN with one-hot state encoding, reward shaping wrappers, and sweep tooling across grid sizes.

### PPO — Proximal Policy Optimization
SB3-based PPO with one-hot encoding, potential-based reward shaping, training callbacks, and a full experiment pipeline. Default tuned config: 8x8 grid, 200k timesteps, MLP 64x64.

### A2C — Advantage Actor-Critic
SB3-based A2C baseline.

## Structure

```
MCTS/           - MCTS implementation and experiments
DQN/            - DQN implementation and sweep scripts
PPO/            - PPO implementation and experiment pipeline
A2C/            - A2C implementation
maps.json       - Shared pre-generated maps (sizes 4–128)
```

## Usage

```bash
# MCTS
cd MCTS
python run_mcts.py --selection uct --rollout random --final robust_child --grid 8 --episodes 100
python run_batch.py   # run all configurations

# PPO
cd PPO
python3  run_ppo.py

# DQN
cd DQN
python3 run_dqn_sweep.py

# A2C
cd A2C
python3 a2c.py
```

## Environment

FrozenLake-v1 on 8x8 grid, with and without slipperiness.

## Requirements

```bash
pip3 install gymnasium stable-baselines3 torch matplotlib tqdm
```

# Supplemental Results
## Table of Contents
1. [DQN: One-Hot vs Discrete Encoding](#1-dqn-one-hot-vs-discrete-encoding)
2. [DQN: Reward Shaping vs Baseline](#2-dqn-reward-shaping-vs-baseline)
3. [DQN: Per-Condition Sweep Results](#3-dqn-per-condition-sweep-results)
4. [MCTS: Average Reward by Component](#4-mcts-average-reward-by-component)
5. [MCTS: Average Steps to Goal by Component](#5-mcts-average-steps-to-goal-by-component)
6. [MCTS: Average Episode Time by Component](#6-mcts-average-episode-time-by-component)
7. [PPO: Optimized Configuration](#7-ppo-optimized-configuration)
8. [PPO: GAE Lambda Variation](#8-ppo-gae-lambda-variation)
9. [PPO: Value Loss Coefficient Variation](#9-ppo-value-loss-coefficient-variation)
10. [PPO: Reward Shaping](#10-ppo-reward-shaping)

---

### 1. DQN: One-Hot vs Discrete Encoding
#### Average Over All Configs
<img src="DQN/dqn_plots/sweep/compare_latest/onehot_vs_discrete_avg_over_configs_map8x8.png" width="600"/>

#### Fixed Config5 (More Explore)
<img src="DQN/dqn_plots/sweep/compare_latest/onehot_vs_discrete_fixed_cfg5_more_explore_map8x8.png" width="600"/>

#### Sweep Mean ± Std (One-Hot)
<img src="DQN/dqn_plots/sweep/compare_latest/sweep_cfg_mean_std_onehot_discrete_200k.png" width="600"/>

---

### 2. DQN: Reward Shaping vs Baseline
#### Average Over All Configs
<img src="DQN/dqn_plots/sweep/compare_latest/shaped_vs_baseline_avg_over_configs_map8x8.png" width="600"/>

#### Best/Worst Per Run
<img src="DQN/dqn_plots/sweep/compare_latest/shaped_vs_baseline_best_worst_per_run_map8x8.png" width="600"/>

#### Fixed Config5 (More Explore)
<img src="DQN/dqn_plots/sweep/compare_latest/shaped_vs_baseline_fixed_cfg5_more_explore_map8x8.png" width="600"/>

---

### 3. DQN: Per-Condition Sweep Results
#### Sweep Mean ± Std (Shaped)
<img src="DQN/dqn_plots/sweep/compare_latest/sweep_cfg_mean_std_shaped_discrete_200k.png" width="600"/>

---

### 4. MCTS: Average Reward by Component
<img src="MCTS/graphs/comparison/avg_reward_overview.png" width="600"/>

---

### 5. MCTS: Average Steps to Goal by Component
<img src="MCTS/graphs/comparison/avg_steps_overview.png" width="600"/>

---

### 6. MCTS: Average Episode Time by Component
<img src="MCTS/graphs/comparison/avg_episode_time_overview.png" width="600"/>

---

### 7. PPO: Optimized Configuration
*lr=1e-3, 200k timesteps, seed 2*

| | | |
|---|---|---|
| <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_training_curve_seed2.png" width="400"/> | <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_episode_length_seed2.png" width="400"/> | <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_approx_kl_seed2.png" width="400"/> |

---

### 8. PPO: GAE Lambda Variation
*All runs at 3 seeds, seed 2 shown*

#### λ=0.99
| Learning Curve | Episode Length | KL Divergence |
|---|---|---|
| <img src="PPO/results/8x8/gae_099/seed2/ppo_training_curve_seed2.png" width="250"/> | <img src="PPO/results/8x8/gae_099/seed2/ppo_episode_length_seed2.png" width="250"/> | <img src="PPO/results/8x8/gae_099/seed2/ppo_approx_kl_seed2.png" width="250"/> |

#### λ=0.95 (baseline)
| Learning Curve | Episode Length | KL Divergence |
|---|---|---|
| <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_training_curve_seed2.png" width="250"/> | <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_episode_length_seed2.png" width="250"/> | <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_approx_kl_seed2.png" width="250"/> |

#### λ=0.85
| Learning Curve | Episode Length | KL Divergence |
|---|---|---|
| <img src="PPO/results/8x8/gae_085/seed2/ppo_training_curve_seed2.png" width="250"/> | <img src="PPO/results/8x8/gae_085/seed2/ppo_episode_length_seed2.png" width="250"/> | <img src="PPO/results/8x8/gae_085/seed2/ppo_approx_kl_seed2.png" width="250"/> |

---

### 9. PPO: Value Loss Coefficient Variation

#### vf_coef=0.25
| Learning Curve | Episode Length | KL Divergence |
|---|---|---|
| <img src="PPO/results/8x8/vf_coef_025/seed2/ppo_training_curve_seed2.png" width="250"/> | <img src="PPO/results/8x8/vf_coef_025/seed2/ppo_episode_length_seed2.png" width="250"/> | <img src="PPO/results/8x8/vf_coef_025/seed2/ppo_approx_kl_seed2.png" width="250"/> |

#### vf_coef=0.5 (baseline)
| Learning Curve | Episode Length | KL Divergence |
|---|---|---|
| <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_training_curve_seed2.png" width="250"/> | <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_episode_length_seed2.png" width="250"/> | <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_approx_kl_seed2.png" width="250"/> |

#### vf_coef=1.0 (3 seeds — appears promising)
| Learning Curve | Episode Length | KL Divergence |
|---|---|---|
| <img src="PPO/results/8x8/vf_coef_10/seed2/ppo_training_curve_seed2.png" width="250"/> | <img src="PPO/results/8x8/vf_coef_10/seed2/ppo_episode_length_seed2.png" width="250"/> | <img src="PPO/results/8x8/vf_coef_10/seed2/ppo_approx_kl_seed2.png" width="250"/> |

#### vf_coef=1.0 (5-seed verification — variance increase)
| Learning Curve | Episode Length | KL Divergence |
|---|---|---|
| <img src="PPO/results/8x8/vf_coef_10_5seeds/seed2/ppo_training_curve_seed2.png" width="250"/> | <img src="PPO/results/8x8/vf_coef_10_5seeds/seed2/ppo_episode_length_seed2.png" width="250"/> | <img src="PPO/results/8x8/vf_coef_10_5seeds/seed2/ppo_approx_kl_seed2.png" width="250"/> |

---

### 10. PPO: Reward Shaping

#### No Shaping (optimized baseline)
| Learning Curve | Episode Length | KL Divergence |
|---|---|---|
| <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_training_curve_seed2.png" width="250"/> | <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_episode_length_seed2.png" width="250"/> | <img src="PPO/results/8x8/lr1e3_200k/seed2/ppo_approx_kl_seed2.png" width="250"/> |

#### Step Penalty=0.01
| Learning Curve | Episode Length | KL Divergence |
|---|---|---|
| <img src="PPO/results/8x8/step_penalty_001/seed2/ppo_training_curve_seed2.png" width="250"/> | <img src="PPO/results/8x8/step_penalty_001/seed2/ppo_episode_length_seed2.png" width="250"/> | <img src="PPO/results/8x8/step_penalty_001/seed2/ppo_approx_kl_seed2.png" width="250"/> |

#### PBRS Manhattan=0.01
| Learning Curve | Episode Length | KL Divergence |
|---|---|---|
| <img src="PPO/results/8x8/manhattan_001/seed2/ppo_training_curve_seed2.png" width="250"/> | <img src="PPO/results/8x8/manhattan_001/seed2/ppo_episode_length_seed2.png" width="250"/> | <img src="PPO/results/8x8/manhattan_001/seed2/ppo_approx_kl_seed2.png" width="250"/> |
