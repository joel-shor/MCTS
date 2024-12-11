"""MCTS search class.

To test:

    python -m mcts.searcher.mcts

"""

from enum import Enum
import math
import random
import time

from mcts.base import base


def random_policy(state: base.BaseState) -> float:
    while not state.is_terminal():
        try:
            action = random.choice(state.get_possible_actions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.take_action(action)
    return state.get_reward()


class TreeNode:
    def __init__(self, state, parent):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        s = ["totalReward: %s" % self.totalReward,
             "numVisits: %d" % self.numVisits,
             "isTerminal: %s" % self.is_terminal,
             "possibleActions: %s" % (self.children.keys())]
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))
    
    def all_visits(self) -> int:
        """Returns number of visits of all nodes in the tree, including this one."""
        if self.is_terminal:
            return self.numVisits
        return sum([x.all_visits() for x in self.children.values()]) + self.numVisits


class BackpropMethod(Enum):
    """Determines how backpropagated rewards are calculated.
    
    `AVERAGE` is the default: sum_all_children / num_visited
    `MAX` uses the max child node: max_child
    """
    AVERAGE = 1
    MAX = 2


class MCTS:
    """Stateful MCTS class.
    
    Typical usage pattern should be as follows:

    ```python
    mcts_game = MCTS()
    mcts_game.reset_game(initial_state)

    while not mcts.root.state.is_terminal():
        action = mcts_game.search()
        mcts_game.take_action(action)
    ```
    """
    def __init__(self,
                 time_limit: int = None,
                 iteration_limit: int = None,
                 exploration_constant: float = math.sqrt(2),
                 rollout_policy=random_policy,
                 backprop_method: BackpropMethod = BackpropMethod.AVERAGE):
        self.root = None
        if time_limit is not None:
            if iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = time_limit
            self.limit_type = 'time'
        else:
            if iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.search_limit = iteration_limit
            self.limit_type = 'iterations'
        self.exploration_constant = exploration_constant
        self.rollout_policy = rollout_policy
        self.backprop_method = backprop_method

    def reset_game(self, initial_state: base.BaseState) -> None:
        """Resets the game, starts with a state."""
        self.root = TreeNode(initial_state, None)

    def search(self, need_details: bool = None):
        if self.root is None:
            raise ValueError(f'Game must be reset before calling search().')
        if self.root.state is None:
            raise ValueError(f'Game must be reset before calling search().')

        if self.limit_type == 'time':
            time_limit = time.time() + self.timeLimit / 1000
            while time.time() < time_limit:
                self.execute_round()
        else:
            for i in range(self.search_limit):
                self.execute_round()

        best_child = self.get_best_child(self.root, 0)
        action = (action for action, node in self.root.children.items() if node is best_child).__next__()
        if need_details:
            if self.backprop_method == BackpropMethod.AVERAGE:
                best_reward = best_child.totalReward / best_child.numVisits
            elif self.backprop_method == BackpropMethod.MAX:
                best_reward = best_child.totalReward
            else:
                raise Exception("Should never reach here")
            return action, best_reward
        else:
            return action
        
    def take_action(self, action: base.BaseAction) -> None:
        """Takes an action and reuses the existing tree."""
        action_child_dict = dict(self.root.children.items())
        if action not in action_child_dict:
            raise ValueError(f"Action not in possible actions: {action} vs {action_child_dict.keys()}")
        self.root = action_child_dict[action]
        # Kill the parent, for memory reasons.
        del self.root.parent
        self.root.parent = None

    def execute_round(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.select_node(self.root)
        reward = self.rollout_policy(node.state)
        self.backpropogate(node, reward)

    def select_node(self, node: TreeNode) -> TreeNode:
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                return self.expand(node)
        return node

    def expand(self, node: TreeNode) -> TreeNode:
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children:
                newNode = TreeNode(node.state.take_action(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node: TreeNode, reward: float):
        while node is not None:
            node.numVisits += 1
            if self.backprop_method == BackpropMethod.AVERAGE:
                node.totalReward += reward
            elif self.backprop_method == BackpropMethod.MAX:
                node.totalReward = max(node.totalReward, reward)
            else:
                raise ValueError("Unknown backprop method: %s" % self.backprop_method)
            node = node.parent

    def get_best_child(self, node: TreeNode, explorationValue: float, exploration_value: float = None) -> TreeNode:
        exploration_value = explorationValue if exploration_value is None else exploration_value
        best_value = float("-inf")
        best_nodes = []
        for child in node.children.values():
            if self.backprop_method == BackpropMethod.AVERAGE:
                exploit = node.state.get_current_player() * child.totalReward / child.numVisits
            elif self.backprop_method == BackpropMethod.MAX:
                exploit = node.state.get_current_player() * child.totalReward
            else:
                raise ValueError("Unknown backprop method: %s" % self.backprop_method)
            explore = math.sqrt(math.log(node.numVisits) / child.numVisits)
            node_value = exploit + exploration_value * explore
            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes)

if __name__ == '__main__':
    # Run some simple tests.
    MCTS(time_limit=1)
    MCTS(time_limit=1, backprop_method=BackpropMethod.MAX)