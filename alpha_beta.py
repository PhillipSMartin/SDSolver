from __future__ import annotations
from ruleset import Conjunction, RuleSet, Disjunction
from time import time
from typing import Any

class Node:
    """
    Base class for game nodes

    Properties:
        state -- current game state
        plays -- list containing each play made so far
        next_to_play -- 0 for max, 1 for min
        solution -- tuple containing the best next play and the resulting score

    General methods:
        id() -> str -- returns unique identifier
        player_names() -> list[str] -- returns list of players' names, default is ['max', 'min']
        player() -> str -- returns name of next player to play
        opponent() -> str -- returns name of player()'s opponent
        show_solution() -> str -- returns a verbal description of the solution
        __str__()->str -- returns a string representation to be used in log messages
        
        is_terminal() -> bool -- returns True for a terminal node

        make_play(play) -- updates self.plays to reflect specified play
            calls compute_state() and updates self.state
            calls compute_score and updates self.score if play results in a terminal node
        compute_state(play) -- calculates new state based on current state and specified play
        compute_score() -> float or None -- calculates the score of a terminal node

        get_options() -- returns a list of possible plays
        get_child(play) -- returns a child of this node resulting from making the specified play
        get_children() -- returns a list of child nodes, one for each play returned by get_options()
        get_state() -- returns a copy of self.state
        get_plays() -- returns a copy of self.plays
        get_last_play() -- returns the most recent play

        evaluate() -> float or str -- returns score of a terminal node
            returns string representing next to play for a non-terminal node

    Methods required to use alpha_beta_partitioned search:
        is_similar_rule(node) -- determines if specified node is similar to this node
        can_precede_rule(node, rule_set) -- determines if a successor to the specified node is in the specified set
        can_precede_only_rule(node, rule_set) -- determines if all successors to the specified node are in the specified set
        find_play_to_reach(rule_set) -- returns a play needed to move from this node to some node in the specified set

        is_similar() -- returns a RuleSet that implements is_similar_rule()
        can_precede(rule_set) -- returns a RuleSet that implements can_precede_rule
        can_precede_only(rule_set) -- returns a RuleSet that implements can_precede_only_rule

    """

    def __init__(self, parent=None, play=None, next_to_play=None, default_state=None, default_plays=None):
        """
        initializer

        :param parent: immediate precursor of this node
        :param play: the play that produces this node from the parent
        :param next_to_play: 0 if max plays next; 1 if min; Defaults to opposite of parent or max if no parent
        """
        if default_plays is None:
            default_plays = []
        self.state = parent.get_state() if parent else default_state or ''
        self.plays = parent.get_plays() if parent else default_plays or []
        self.next_to_play = next_to_play or (1 ^ parent.next_to_play if parent else 0)
        self.solution = (None, None)

        if play is not None:
            # updates state and plays
            # updates solution if this is a terminal node
            self.make_play( play )

    def id(self) -> str:
        """ returns unique identifier """
        return self.state

    @staticmethod
    def player_names():
        """ returns list of players' names """
        return ['max', 'min']

    def player(self):
        """ returns name of next player to play """
        return self.player_names()[self.next_to_play]

    def opponent(self):
        """ returns name of player()'s opponent """
        return self.player_names()[self.next_to_play ^ 1]

    def show_solution(self) -> str:
        """ returns a verbal description of the solution """
        play, score = self.solution
        if self.is_terminal() :
            return f'Terminal node, score={score}'
        elif play is not None:
            return  f'{self.player()} can play {play} to guarantee a score of {score}'
        else:
            return 'Not solved'

    def __str__(self) -> str:
        """ returns a string representation to be used in log messages """
        return self.id() if len(self.id()) > 0 else 'root'

    def is_terminal(self) -> bool:
        """  returns True for a terminal node  - must be overridden """
        return False

    def make_play(self, play):
        """
        updates self.plays to reflect specified play
            calls compute_state() and updates self.state
            calls compute_score and updates self.solution if play results in a terminal node

        :param play: the play that produces this node from the parent
        """
        self.plays.append(play)
        self.state = self.compute_state(play)
        self.solution = (None, self.compute_score())

    def compute_state(self, play):
        """
        calculates new state based on current state and specified play
        default implementation is to create a string from the list of plays

        :param play: the play that produces this node from the parent
        :return: new state
        """
        return "".join(self.plays)

    def compute_score(self) -> float or None:
        """
        calculates the score of a terminal node
        must be overridden

        :return: score for a terminal node; None for a non-terminal node
        """
        return 0 if self.is_terminal() else None

    def get_options(self):
        """
        returns a list of possible plays
        must be overridden

        nodes should be returned in opposite order of the order you want to evaluate them in
        """
        return []

    def get_child(self, play):
        """
        returns a child of this node resulting from making the specified play
        must be overridden
        """
        return Node(parent=self, play=play)

    def get_children(self):
        """ returns a list of child nodes, one for each play returned by get_options() """
        return [] if self.is_terminal() else [ self.get_child(option) for option in self.get_options() ]

    def get_state(self):
        """  returns a copy of self.state  - must be overridden if state is not a string"""
        return self.state

    def get_plays(self):
        """ returns a copy of self.plays - must be overridden if plays is not copyable"""
        return self.plays.copy()

    def get_last_play(self):
        """ returns the most recent play - must be overridden if plays is not a list"""
        return None if len(self.plays) == 0 else self.plays[-1]

    def evaluate(self) -> float or None:
        """
            returns score of a terminal node
            returns string representing next to play for a non-terminal node
        """
        return self.solution[1] if self.is_terminal() else self.player()

    def is_similar_rule(self, node:Node) -> bool:
        """
        determines if specified node is similar to this node
        the default rule: terminal nodes are similar if they have the same score;
        non-terminal nodes are similar only to themselves
        
        :param node: node to test
        :return: True if similar, else False
        """
        if self.is_terminal() and node.is_similar():
            return self.compute_score() == node.compute_score()
        else:
            return self.id() == node.id()

    def is_similar(self) -> RuleSet:
        return RuleSet(self.is_similar_rule, f"nodes similar to {self}")

    def can_precede_rule(self, node:Node, rule_set:RuleSet) -> bool:
        """
        determines if a successor to the specified node is in the specified set

        :param node: the node to consider
        :param rule_set: the set of possible successors
        :return: True if set contains a successor to the specified node, else False
        """
        return any(child in rule_set for child in node.get_children())

    def can_precede(self, rule_set:RuleSet, descr:str=None)->RuleSet:
        return RuleSet(lambda node: self.can_precede_rule(node, rule_set), descr=descr)

    def can_precede_only_rule(self, node: Node, rule_set:RuleSet) -> bool:
        """
        determines if all successors to the specified node are in the specified set

        :param node: the node to consider
        :param rule_set: the set of possible successors
        :return: rue if set contains all successors to the specified node, else False
        """
        return all(child in rule_set for child in node.get_children())
    
    def can_precede_only(self, rule_set:RuleSet, descr:str=None)->RuleSet:
        return RuleSet(lambda node: self.can_precede_only_rule(node, rule_set), descr=descr)

    def find_play_to_reach(self, rule_set:RuleSet):
        """
        returns a play needed to move from this node to some node in the specified set

        :param rule_set: the set we wish to reach
        :return: a play to reach it
        """
        for child in self.get_children():
            if child in rule_set:
                return child.plays[-1]
        return None


class TestNode(Node):
    """
    Node for a simple game where each player plays either 'A' or 'B'
    The final score for max is the number of A's played

    Overrides:
        __init__(max_plays)
        is_terminal()
        compute_score()
        get_options()
        get_child()
    """

    max_plays = 2

    def __init__(self, parent=None, play=None, next_to_play=None, max_plays:int = None):
        """
        Initializer

        :param parent: immediate precursor of this node
        :param play: the play that produces this node from the parent
        :param next_to_play: 0 if max plays next; 1 if min; Defaults to opposite of parent or max if no parent
        :param max_plays: defines number of plays before a note is declarer terminal, default is 2
            (needed only for root)
        """
        if max_plays:
            TestNode.max_plays = max_plays
        super().__init__(parent=parent, play=play, next_to_play=next_to_play)

    def is_terminal(self) -> bool:
        """  returns True for a terminal node """
        return len(self.plays) >= TestNode.max_plays

    def compute_score(self) -> float or None:
        """
        calculates the score of a terminal node

        :return: number of A's played for a terminal node; None for a non-terminal node
        """
        if not self.is_terminal():
            return None
        else:
            score = self.id().count('A')
            return score if self.next_to_play == 0 else -score

    def get_options(self):
        """ returns a list of possible plays """
        return [] if self.is_terminal() else ['B', 'A']

    def get_child(self, play):
        """ returns a child of this node resulting from making the specified play  - must be overridden """
        return TestNode(parent=self, play=play)

class NTransposition:
    """
    for saving the solution for a given node, alpha, and beta
    """

    def __init__(self, node:Node, alpha:float, beta:float, solution=(Any, float)):
        self.key = (node.id(), alpha, beta)
        self.value = solution

    def is_hit(self, node, alpha, beta):
        return self.key == (node.id(), alpha, beta)

    def __str__(self):
        return f'{self.key[0]}, alpha={self.key[1]}, beta={self.key[2]}, solution={self.value}'

# for saving the score associated with a set of nodes, given a specific alpha and beta
class STransposition:
    """
    for saving the solution for a given set of similar nodes, alpha, and beta
    """

    def __init__(self, problem_set:RuleSet, alpha:float, beta:float, value:(RuleSet or set, float)=(set(), None)):
        self.problem_set = problem_set
        self.limits = (alpha, beta)
        self.value = value

    def is_hit(self, node, alpha, beta):
        return node in self.problem_set and self.limits == (alpha, beta)

    def __str__(self):
        return f'{self.problem_set}, alpha={self.limits[0]}, beta={self.limits[1]}, score={self.value[1]}, solution_set={self.value[0]}'

class TranspositionTable:
    def __init__(self):
        self.table = []

    # returns the entry for a specific node, alpha, and beta
    # returns None if not in the table
    def find(self, node, alpha, beta):
        for entry in self.table:
            if entry.is_hit(node, alpha, beta):
                return entry
        return None

    # add an entry to the table
    def add(self, item):
        self.table.append(item)

class Stats:
    def __init__(self, max_levels):
        self._nodes_evaluated = [0] * (max_levels + 1)
        self._prunes = [0] * (max_levels + 1)
        self._table_hits = [0] * (max_levels + 1)
        self._start_time = time()
        self._execution_time = 0

    def clear(self):
        self._nodes_evaluated[:] = [0] * len(self._nodes_evaluated)
        self._prunes[:] = [0] * len(self._prunes[:])
        self._table_hits[:] = [0] * len(self._table_hits[:])
        self._start_time = time()
        self._execution_time = 0

    def end(self):
        self._execution_time = time() - self._start_time

    def __str__(self):
        return f'Total nodes evaluated = {sum(self._nodes_evaluated)}' \
            f'\nNodes evaluated per level = {self._nodes_evaluated}' \
            f'\nTotal prunes = {sum(self._prunes)}' \
            f'\nPrunes per level = {self._prunes}' \
            f'\nTable hits per level = {self._table_hits}' \
            f'\nExecutiontime = {self._execution_time * 1000:.0f} ms'

    def node_evaluated(self, level):
        self._nodes_evaluated[level] += 1

    def branch_pruned(self, level):
        self._prunes[level] += 1

    def table_hit(self, level):
        self._table_hits[level] += 1

class Game:
    log_message_number = 0

    def __init__(self, min_score = 0.0, max_score = 1.0, max_levels = 6):
        self.min_score = min_score
        self.max_score = max_score
        self.max_levels = max_levels
        self.t_table = TranspositionTable()
        self.stats = Stats(max_levels)

    def solve(self, node:Node, alpha:float, beta:float, partitioned=False, verbose=False):
        self.stats.clear()
        if not partitioned:
            self.alpha_beta(node, alpha, beta, verbose=verbose)
        else:
            self.alpha_beta_partitioned(node, alpha, beta, verbose=verbose)
        print(f'SOLUTION:  {node}: {node.show_solution()}')
        self.stats.end()
        return node

    def play(self, node:Node, alpha:float, beta:float, partitioned=False, verbose=False):
        play = input(f"Enter play for {node.player()}, or hit enter to have program play. Enter 'q' to quit: ")
        while play != 'q':
            if not play:
                self.solve(node, alpha, beta, partitioned=partitioned, verbose=verbose)
                play = node.solution[0] or 'q'
            else:
                node = node.get_child(play)
                play = input(f"CURRENT NODE: {node} - Enter play for {node.player()}, or hit enter to have program play. Enter 'q' to quit: ")

    def show_stats(self):
        print(self.stats)

    def alpha_beta(self, node:Node, alpha:float, beta:float, level=0, verbose=False):
        """
        conducts an alpha-beta search

        :param node: current node
        :param alpha:  max has a way to guarantee this score within this branch
        :param beta: min has a way to hold max to this score within this branch
        :param level: starts at 0, incremented as we progress along the tree
        :param verbose: for tracing logic

        :return: the solution: a tuple consisting of the best play and the resulting score
        """

        if verbose:
            self.log(f'SOLVING {node} ({node.player()} to play), alpha={alpha}, beta={beta}', level)

        # see if we've seen this problem before
        hit = None if node.is_terminal() else self.t_table.find(node, alpha, beta)

        # if so, return previously calculated score
        if hit:
            self.stats.table_hit(level)
            if verbose:
                self.log(f'***** Table hit: {hit} *****', level)
            node.solution = hit.value
            _, score = node.solution

        # if not, evaluate it
        else:
            # value is
            #   the name of player who is next to play for a non-terminal node
            #   the score for a terminal node
            value = node.evaluate()

            # non-terminal node
            if isinstance(value, str):
                score = self.expand_node(node, alpha, beta, level=level, verbose=verbose)
                self.t_table.add(NTransposition(node, alpha, beta, node.solution))

            # terminal node
            # score should be minus if we are solving for min
            else:
                score = value if alpha >= 0 else -value

            self.stats.node_evaluated(level)

        # solved
        if verbose:
            self.log(f'SOLVED {node} ({node.player()} to play), alpha={alpha}, beta={beta}: {node.show_solution()}', level)
        return -score

    def alpha_beta_partitioned(self, node:Node, alpha:float, beta:float, level=0, verbose=False):
        """
        conducts an alpha-beta partitioned search

        :param node: current node
        :param alpha:  max has a way to guarantee this score
            if examining plays for min, once he has a way to hold max to this score,
            he can't do better, so there is no need to continue our examination
        :param beta: min has a way to hold max to this score
            if examining plays for max, once he has a way to guarantee this score,
            he can't do better, so there is no need to continue our evaluation
        :param level: starts at 0, incremented as we progress along the tree
        :param verbose: for tracing logic

        :return: a tuple, consisting of
            the problem set: a set of nodes similar to the node being solved
            the solution: a tuple consisting of the best play and the resulting score
        """

        if verbose:
            self.log(f'SOLVING {node} ({node.player()} to play), alpha={alpha}, beta={beta}', level)

        # see if we've seen this problem before
        hit = None if node.is_terminal() else self.t_table.find(node, alpha, beta)

        # if so, return previously calculated solution
        if hit:
            self.stats.table_hit(level)

            # hit.problem set is the set of similar problems this node was found to be a member of
            # hit.solution will be a tuple, consisting of the solution set and the score
            # we must calculate the play that will yield some node in the solution set
            problem_set = hit.problem_set
            solution_set, score = hit.value
            best_play = node.find_play_to_reach(solution_set)
            if verbose:
                self.log(f'***** Table hit: {hit}, play={best_play} *****', level)

            node.solution = (best_play, score)

        # if not, evaluate it
        else:
            # value is
            #   the name of player who is next to play for a non-terminal node
            #   the score for a terminal node
            value = node.evaluate()

            if isinstance(value, str):
                problem_set, solution_set = self.expand_node_partitioned(node, alpha, beta, level=level, verbose=verbose)
                _, score = node.solution
                self.t_table.add(STransposition(problem_set, alpha, beta, (solution_set, score)))
            else:
                problem_set = node.is_similar()
                # score should be minus if we are solving for min
                _, score = node.solution
                if alpha < 0:
                    score *= -1

            self.stats.node_evaluated(level)

        # solved
        if not node in problem_set:
            raise Exception(f'Problem set does not contain the specified problem node')

        # return problem_set and solution
        if verbose:
            self.log(f'SOLVED {node} ({node.player()} to play), alpha={alpha}, beta={beta}: {node.show_solution()}', level)
        return problem_set, -score

    def expand_node(self, node: Node, alpha: float, beta: float, level: int, verbose: bool):
        """
        find solution node among children of the specified node

        :param node: the node we are solving
        :param alpha:  player already has a way to guarantee this score (negative for min)
        :param beta: opponent has a way to hold player to this score  (negative for min)
        :param level:
        :param verbose:
        :return: best score (negative for min)
        """

        best_play = None
        best_score = float('-inf')

        #  calculate score for each child node
        children = node.get_children().copy()
        while len(children) > 0:
            child = children.pop()
            if verbose:
                self.log(f'Considering {node}: {node.player()} plays {child.get_last_play()}', level)

            new_score = self.alpha_beta(child, - beta, - max(best_score, alpha), level=level + 1, verbose=verbose)

            # if new solution improves on previous solution, save it
            if new_score > best_score:
                if verbose:
                    self.log(
                        f'Result for {node}: {node.player()} plays {child.get_last_play()}: Selecting, since its score of {new_score} improves on previous best score of {best_score}',
                        level)
                best_play, best_score = (child.get_last_play(), new_score)

                # if this play achieves beta, skip remaining children
                if new_score >= beta and len(children) > 0:  # make sure we have children to prune
                    self.stats.branch_pruned(level + 1)
                    if verbose:
                        self.log(
                            f"Pruning {[child.id() for child in children]}, since {node.player()} can't do better than {beta}",
                            level)
                    break
            else:
                if verbose:
                    self.log(
                        f'Result for {node}: {node.player()} plays {child.get_last_play()}: Rejecting, since its score of {new_score} does not improve on previous best score of {best_score}',
                        level)

        # we have found the solution
        node.solution = (best_play, best_score)
        return best_score

    def expand_node_partitioned(self, node:Node, alpha:float, beta: float, level:int, verbose:bool):
        """
        find solution node among children of the specified node

        :param node: the node we are solving
        :param beta:
        :param alpha:
        :param level:
        :param verbose:
        :return: problem set and solution set
            (node will be updated with solution)
        """

        # a tuple of the best play and the resulting score
        best_play = None
        best_score = float('-inf')
        # the set of nodes similar to the solution node
        solution_set = RuleSet()
        # the set of all solution sets we have examined
        solution_set_pool = Disjunction(descr=f'solution set pool for {node}')

        #  calculate score for each child node
        children = node.get_children().copy()
        while len(children) > 0:
            child = children.pop()
            if verbose:
                self.log(f'Considering {node}: {node.player()} plays {child.get_last_play()}', level)

            # adjust alpha and beta and solve this node
            # new_solution_set is the set of nodes similar to this child node
            # new_solution is (best_play, score)
            new_solution_set, new_score = self.alpha_beta_partitioned(child, -beta, -max(best_score, alpha), level=level + 1, verbose=verbose)

            # keep track of nodes similar to all nodes we have examined
            solution_set_pool.add_rule_set(new_solution_set)

            # if new solution improves on previous solution, save it
            if new_score > best_score:
                if verbose:
                    self.log(f'Result for {node}: {node.player()} plays {child.get_last_play()}: Selecting, since its score of {new_score} improves on previous best score of {best_score}', level)
                best_play, best_score = (child.get_last_play(), new_score)
                solution_set = new_solution_set

                # if this play achieves beta for max or alpha for min, skip remaining children
                if best_score >= beta and len(children) > 0: # make sure we have children to prune
                    self.stats.branch_pruned(level + 1)
                    if verbose:
                        self.log(f"Pruning {[child.id() for child in children]}, since {node.player()} can't do better than {beta}", level)

                    # we won't have nodes similar to pruned nodes, but we add the child
                    #  itself as the next best thing
                    for child in children:
                        solution_set_pool.add_rule_set(child.is_similar())
                    break
            else:
                if verbose:
                    self.log(f'Result for {node}: {node.player()} plays {child.get_last_play()}: Rejecting, since its score of {new_score} does not improve on previous best score of {best_score}', level)

        # we have found the solution
        node.solution = (best_play, best_score)

        # construct the problem set, i.e. nodes similar
        #   to this node that have the same solution
        if best_score == self.min_score:
            # Since we can't score from this position, we won't be able to score from any node constrained to our universe
            problem_set = node.can_precede_only(solution_set_pool, descr=f'Nodes similar to {node}')
        else:
            # Nodes similar to our problem node belong to the intersection of two sets:
            #   Nodes that can reach solution_set
            #   Nodes constrained to reach solution_set_pool
            problem_set = Conjunction(node.can_precede(solution_set), node.can_precede_only(solution_set_pool), descr=f'Nodes similar to {node}')

        return problem_set, solution_set

    @staticmethod
    def log(msg:str, level:int):
        Game.log_message_number += 1
        print(f"{Game.log_message_number} {'.' * level * 2}{msg}")

        # if Game.log_message_number == 30:
        #     print('here')
        if Game.log_message_number > 20753 :
            quit()

if __name__ ==  '__main__':
    root = TestNode(max_plays = 6)
    game = Game(min_score = 0.0, max_score = 6.0, max_levels = 6)
    game.solve(root,0, 6, partitioned=False, verbose=True)
    # game.play(node,0, 6, partitioned=False, verbose=True)
    game.show_stats()








