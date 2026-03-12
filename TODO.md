# PPO TODO

## Critical

### ent_coef was 0.01, now corrected to 0.0
Previous runs used ent_coef=0.01 (from the A2C config in the proposal,
not PPO) - SB3 PPO default is 0.0 - all previous results need to be
re-run with ent_coef=0.0

### Net arch sizes [2, 4, 8, 16] are NOT SB3 baselines
Dragan et al chose these small sizes specifically for fair comparison
with quantum circuits that have few trainable parameters - SB3 default
is [64, 64] - we should run hidden=64 as our true SB3 baseline and
treat 2/4/8/16 as the Dragan et al comparison configs

### Convergence metric reports "converged" for failed agents
The Dragan et al definition (reward stays within 0.2 for all future
points) triggers for agents that flatline near zero - hidden=4 and
hidden=2 both "converged" at 10,000 timesteps because they never
learned anything - need to add minimum reward threshold or report
convergence as (timestep, reward_at_convergence)

### Compute map-specific optimal policy
Currently using 0.5 as sample efficiency threshold - Dragan et al
used 0.81 based on optimal policy for their specific map - we need
to solve for optimal policy on our seed=42 map using value iteration
to get our own threshold

## Investigation Needed

### Hidden=4 gets 0% success
16->4->4->4 bottleneck too narrow for 16-dim one-hot input
Check: different seeds, more timesteps, without one-hot encoding

### Hidden=2 eval (57.7%) doesn't match training peak (0.200)
Deterministic eval may luck into a working path on this specific map
Need multiple seeds to confirm

## Runs Needed
- [ ] Re-run all hidden sizes with ent_coef=0.0
- [ ] Run hidden=64 as true SB3 baseline
- [ ] Run multiple seeds and average results

## Configurations to Test (from proposal)
- Advantage functions
- Exploration strategies
- Reward shaping (use reward_processing hook in ModPPO)
- Loss function variations
- Hyperparameter optimization

## Stretch
- Two-phase training for stochastic mode (from FrozenLake PPO repo)
- Compare against Dragan et al classical baseline numbers directly