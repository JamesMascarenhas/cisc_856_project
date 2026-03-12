"""
ModPPO - Modified Proximal Policy Optimization built on Stable-Baselines3

Extends SB3 PPO with hooks for:
  - Reward shaping (override `reward_processing`)
  - Exploration strategies (override `action_selection`)

The train() method matches upstream SB3 PPO exactly (Raffin et al, 2021)
so results are directly comparable to the classical baseline

SB3 PPO defaults (used unless overridden):
  - learning_rate: 3e-4
  - n_steps: 2048
  - batch_size: 64
  - n_epochs: 10
  - gamma: 0.99
  - gae_lambda: 0.95
  - clip_range: 0.2
  - ent_coef: 0.0
  - vf_coef: 0.5
  - net_arch: [64, 64] (two hidden layers, 64 neurons each)

Note: Dragan et al (2022) used smaller net_arch sizes [2, 4, 8, 16]
specifically for fair comparison with quantum circuits that have few
trainable parameters - these are NOT standard PPO baseline sizes
"""

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class ModPPO(PPO):
    """
    Extended PPO that keeps full SB3 training logic intact while exposing
    hooks for experimentation (reward shaping, exploration strategies)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Training - matches SB3 PPO exactly
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer

        Follows the SB3 PPO implementation:
        1) Set policy to training mode
        2) Update learning rate schedule
        3) Resolve clip_range (may be a schedule)
        4) Loop over n_epochs x mini-batches from rollout buffer
        5) Compute clipped surrogate loss, value loss, entropy loss
        6) Back-propagate and clip gradients
        7) Optional early stopping via target_kl
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                # FrozenLake is Discrete - SB3 stores actions as float
                # but evaluate_actions needs long for discrete spaces
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Advantage normalization
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO Clipped Surrogate Objective
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss (encourages exploration)
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Combined loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Approximate KL for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Logging
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    # Extension hooks - override in subclasses for experiments
    def reward_processing(self, rewards, obs, dones):
        """Reward shaping hook - default: no modification (SB3 baseline)"""
        return rewards

    def action_selection(self, actions, obs):
        """Exploration strategy hook - default: no modification (standard PPO)"""
        return actions

    def select_action(self, observation, deterministic=False):
        """Wrapper around SB3 predict() that returns a plain int"""
        action, _state = self.predict(observation, deterministic=deterministic)
        return int(action)