"""MCTS search class.

To test:

    python -m mcts.searcher.mcts

"""

from enum import Enum
import math
import random
import time

from mcts.base.base import BaseState


def random_policy(state: BaseState) -> float:
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


class BackpropMethod(Enum):
    """Determines how backpropagated rewards are calculated.
    
    `AVERAGE` is the default: sum_all_children / num_visited
    `MAX` uses the max child node: max_child
    """
    AVERAGE = 1
    MAX = 2


class MCTS:
    def __init__(self,
                 time_limit: int = None,
                 timeLimit=None,
                 iteration_limit: int = None,
                 iterationLimit=None,
                 exploration_constant: float = None,
                 explorationConstant=math.sqrt(2),
                 rollout_policy=None,
                 rolloutPolicy=random_policy,
                 backprop_method: BackpropMethod = BackpropMethod.AVERAGE):
        # backwards compatibility
        time_limit = timeLimit if time_limit is None else time_limit
        iteration_limit = iterationLimit if iteration_limit is None else iteration_limit
        exploration_constant = explorationConstant if exploration_constant is None else exploration_constant
        rollout_policy = rolloutPolicy if rollout_policy is None else rollout_policy

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

    def search(self, initialState: BaseState = None, initial_state: BaseState = None, needDetails: bool = False,
               need_details: bool = None):
        initial_state = initialState if initial_state is None else initial_state
        need_details = needDetails if need_details is None else need_details
        self.root = TreeNode(initial_state, None)

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
    
    def all_visits(self) -> int:
        """Returns number of visits of all nodes in the tree, including this one."""
        if self.is_terminal:
            return self.NumVisits
        return sum([x.all_visits() for x in self.children.values()]) + self.numVisits

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