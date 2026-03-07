import gymnasium as gym
import numpy as np
import math
from copy import deepcopy


class Node:
    """Represents a single node in the MCTS search tree."""

    def __init__(self, state, parent, action):
        self.state = state
        self.parent: Node = parent
        self.action = action
        self.children: list[Node] = []
        self.visits = 0  # Number of times this node was visited
        self.value = 0.0  # Total value (reward) accumulated from simulations passing through this node
        self.untried_actions = []
        self.done = False  # Whether this node represents a terminal state

    def is_fully_expanded(self):
        # If no untried actions remain, this node is fully expanded
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return self.done

    def ucb1_score(self, exploration_constant):
        # UCB1 = (value / visits) + exploration_constant * sqrt(ln(parent.visits) / visits)
        if self.visits == 0:
            return float("inf")
        score = (self.value / self.visits) + exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return score

    def best_child(self, exploration_constant) -> "Node":
        max_score = float("-inf")
        best_child = None  # Return self if no children (should not happen if called correctly)
        for child in self.children:
            score = child.ucb1_score(exploration_constant)
            if score > max_score:
                max_score = score
                best_child = child
        return best_child


class MCTS:
    """Monte Carlo Tree Search algorithm"""

    def __init__(self, env: gym.Env, num_simulations, exploration_constant, max_rollout_depth):
        self.env = env
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.max_rollout_depth = max_rollout_depth

    def search(self, state):
        """
        Run the full MCTS algorithm from the given state and return the best action.

        This is the main entry point. It creates the root node, runs
        num_simulations iterations of select -> expand -> rollout -> backpropagate,
        then returns the action leading to the most-visited child.

        Args:
            state: The current environment state to search from.

        Returns:
            The best action to take from the current state.
        """
        pass

    def select(self, node: Node) -> Node:
        current_Node = node
        # Find a leaf node to expand: keep selecting the best child until we find a node that is not fully expanded or is terminal
        while current_Node.is_fully_expanded() and not current_Node.is_terminal():
            best_node = current_Node.best_child(self.exploration_constant)

            if best_node is None:
                break  # No children, return the current leaf node
            current_Node = best_node
        return current_Node

    def expand(self, node: Node) -> Node:
        # Pick an untried action
        action = node.untried_actions.pop()
        cloned_env = self.clone_env_state(node.state)

        # Simulate the action in the cloned environment
        next_state, reward, done, _, _ = cloned_env.step(action)
        child_node = Node(state=next_state, parent=node, action=action)

        # Initialize the child node's untried actions
        child_node.untried_actions = list(range(self.env.action_space.n))

        return child_node

    def rollout(self, node: Node) -> float:
        cloned_env = self.clone_env_state(node.state)
        # Cumulate rewards until a terminal state is reached or max rollout depth is hit
        cumulative_reward = 0.0
        for _ in range(self.max_rollout_depth):
            action = cloned_env.action_space.sample()
            _, reward, done, _, _ = cloned_env.step(action)
            cumulative_reward += reward
            if done:
                break
        return cumulative_reward

    def backpropagate(self, node: Node, reward: float):
        # Propagate the reward up the tree
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def get_action_probabilities(self, root):
        """
        Compute action probabilities based on visit counts of the root's children.

        After all simulations, calculate the probability of each action
        proportional to how many times each child was visited. This can
        be used for training or for stochastic action selection.

        Args:
            root: The root MCTSNode after search is complete.

        Returns:
            A numpy array of action probabilities over the action space.
        """
        pass

    def clone_env_state(self, state) -> gym.Env:
        cloned_env = deepcopy(self.env)
        cloned_env.reset()
        # overwrites the state to the specific one you want
        cloned_env.env.state = state
        return cloned_env
