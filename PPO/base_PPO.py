from stable_baselines3 import PPO
import torch as th

# This file shows the structure to create an agent using PPO as base
# PPO extends A2C with a clipped surrogate objective and multiple
# update epochs per rollout, improving sample efficiency and stability.


class ModPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        # set training mode for policy
        self.policy.set_training_mode(True)

        # PPO iterates over the collected rollout data for multiple epochs
        # (controlled by self.n_epochs), unlike A2C which does a single pass.
        # This improves sample efficiency while the clipping prevents
        # destructively large policy updates.
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                # evaluate actions based on the current policy given the
                # observations and actions from the rollout buffer.
                # returns - estimated value, log likelihood of taking those
                # actions, and entropy of the action distribution.
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                # relative improvement over baseline
                # Advantage = (Reward received + Discounted value of next state) - Value of current state
                advantages = rollout_data.advantages

                # Normalize advantages for better training stability
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # --- PPO Clipped Surrogate Objective ---
                # ratio = π_θ(a|s) / π_θ_old(a|s)
                # Measures how much the new policy deviates from the old one.
                # log_prob comes from the current (updated) policy.
                # rollout_data.old_log_prob comes from the policy that collected the data.
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Unclipped policy loss term (standard policy gradient)
                policy_loss_unclipped = advantages * ratio

                # Clipped policy loss term — clips ratio to [1-ε, 1+ε]
                # Prevents excessively large updates that could destabilize training.
                policy_loss_clipped = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

                # Take the minimum to form a pessimistic lower bound on the objective.
                # The negative sign converts the maximization objective into a loss to minimize.
                policy_loss = -th.min(policy_loss_unclipped, policy_loss_clipped).mean()

                # Value loss
                # Updates the critic (state value estimation).
                # PPO also optionally clips the value function update (self.clip_range_vf).
                if self.clip_range_vf is not None:
                    # Clip value estimates around the old values for stability
                    values_clipped = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                    value_loss_clipped = th.nn.functional.mse_loss(rollout_data.returns, values_clipped)
                    value_loss_unclipped = th.nn.functional.mse_loss(rollout_data.returns, values)
                    # Use the larger of the two losses (conservative estimate)
                    value_loss = th.max(value_loss_unclipped, value_loss_clipped)
                else:
                    value_loss = th.nn.functional.mse_loss(rollout_data.returns, values)

                # Entropy loss
                # High entropy = more random/exploratory actions
                # Low entropy = more deterministic/exploitative actions
                # Negative sign in loss = reward higher entropy
                entropy_loss = -th.mean(entropy)

                # Combined loss with weights (matches proposal: ent_coef=0.01, vf_coef=0.5)
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Backpropagation
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Limits the magnitude of gradients during backpropagation
                # to prevent exploding gradients
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        self._n_updates += self.n_epochs

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        # Rollout collection with reward processing.
        # PPO is on-policy: data is collected under the current policy,
        # used for n_epochs of updates, then discarded.
        rollout_buffer.reset()
        callback.on_rollout_start()

        for step in range(n_rollout_steps):
            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

            # Reward modification — hook for reward shaping experiments
            rewards = self.reward_processing(rewards, new_obs, dones)

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )

            self._last_obs = new_obs
            # If dones is True, the next step is the start of a new episode.
            self._last_episode_starts = dones

        return True

    def reward_processing(self, rewards, obs, dones):
        # Reward shaping logic — modify here for different reward configurations.
        # Baseline: amplify positive rewards, penalise neutral/negative steps.
        return rewards * 2.0 if rewards > 0 else rewards - 0.5

    # Prediction logic with action selection strategy
    def select_action(self, observation, state=None, episode_start=None, deterministic=False):
        # Prediction logic
        self.policy.set_training_mode(False)

        observation, vectorized_env = self.policy.obs_to_tensor(observation)

        with th.no_grad():
            actions = self.policy.get_distribution(observation).get_actions(deterministic=deterministic)

            # Apply action selection strategy (e.g. epsilon-greedy, UCB, Boltzmann)
            if not deterministic:
                actions = self.action_selection(actions, observation)

        actions = actions.cpu().numpy()

        return actions.item()

    def action_selection(self, actions, obs):
        # Exploration strategies during action selection.
        # Override this method to implement custom strategies such as
        # epsilon-greedy, Boltzmann/softmax sampling, or UCB-based selection.
        return actions
