from collections import Counter

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
        get_P() -> set[str]
            if this is a terminal node, returns a set containing the ids of terminal nodes with the same value
            otherwise returns a set containing only this id
        get_R(S) -> set[str]
            returns the ids of nodes like this node that have an immediate successor in set S
        get_C(S) -> set[str]
            returns a set of node ids like this node whose immediate successors are all in set S
        get_play(S)
            returns the play needed to move from this node to some node in the specified set

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
        if self.is_terminal():
            return f'Terminal node, score={self.solution[1]}'

        if not self.solution[0]:
            return 'Not solved'

        return  f'{self.player} can play {self.solution[0]} to guarantee a score of {self.solution[1]}'

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
        return [ self.get_child(option) for option in self.get_options() ]

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

    def get_P(self) -> set[str]:
        """
        if this is a terminal node, returns a set containing the ids of terminal nodes with the same value
        otherwise returns a set containing only this id
        must be overridden
        """
        return { self.id() }

    def get_R(self, S:set) -> set[str]:
        """
        returns nodes that can generate a node in S
        must be overridden

        :param S: a set of similar nodes
        :return: a set of node ids like this node that have an immediate successor in set S
        """
        return { self.id() }

    def get_C(self, S:set):
        """
        returns nodes that must generate a node in S
        must be overridden

        :param S: a set of similar nodes
        :return: a set of node ids like this node whose immediate successors are all in set S
        """
        return { self.id() }

    def get_play(self, S:set):
        """
        returns the play needed to move from this node to some node in the specified set
        must be overridden

        :param S: a set of similar nodes
        """
        return None

MAX_PLAYS = 6
MIN_SCORE = 0.0
MAX_SCORE = float(MAX_PLAYS)

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
        get_P()
        get_R()
        get_S()
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
        super().__init__(parent=parent, play=play, next_to_play=None)

    def is_terminal(self) -> bool:
        """  returns True for a terminal node """
        return len(self.plays) >= TestNode.max_plays

    def compute_score(self) -> float or None:
        """
        calculates the score of a terminal node

        :return: number of A's played for a terminal node; None for a non-terminal node
        """
        return self._counts()[0] if self.is_terminal() else None

    def get_options(self):
        """ returns a list of possible plays """
        return ['B', 'A']

    def get_child(self, play):
        """ returns a child of this node resulting from making the specified play  - must be overridden """
        return TestNode(parent=self, play=play)

    def get_P(self) -> set[str]:
        """
        if this is a terminal node, returns a set containing the ids of terminal nodes with the same value
        otherwise returns a set containing only this id
        """
        if not self.is_terminal():
            return { self.id() }
        else:
            return TestNode._permutations(*self._counts())

    def get_R(self, S:set) -> set[str]:
        """
        returns nodes that can generate a node in S

        :param S: a set of similar nodes
        :return: a set of node ids like this node that have an immediate successor in set S
        """
        return { id[:-1] for id in S }

    def get_C(self, S:set):
        """
        returns nodes that must generate a node in S
        must be overridden

        :param S: a set of similar nodes
        :return: a set of node ids like this node whose immediate successors are all in set S
        """
        counts = Counter([id[:-1] for id in S])
        return {id for id, count in counts.items() if count == 2}

    def get_play(self, S:set):
        """
        returns the play needed to move from this node to some node in the specified set
        must be overridden

        :param S: a set of similar nodes
        """
        for node_id in S:
            if self.id() == node_id[:-1]:
                return node_id[-1]
        return None

    def _counts(self) -> (int, int):
        """
        returns tuple with numbers of A's plays and number of B's played
        """
        return (
            sum([1 for play in self.plays if play == 'A']),
            sum([1 for play in self.plays if play == 'B'])
        )

    @staticmethod
    def _permutations(a_num, b_num) -> set[str]:
        """
        returns all permutations resulting in the specified counts

        :param a_num: number of A's played
        :param b_num: number of B's played
        :return: a set of node ids
        """
        if a_num == 0 and b_num == 0:
            return set()
        if a_num == 0:
            return {'B' * b_num}
        if b_num == 0:
            return {'A' * a_num}
        r = ['A' + item for item in TestNode._permutations(a_num - 1, b_num)]
        r.extend(['B' + item for item in TestNode._permutations(a_num, b_num - 1)])
        return set(r)

class N_Transposition:
    """
    for saving the solution for a given node, alpha, and beta
    """

    def __init__(self, node_id:str, alpha:float, beta:float, solution=(None, None)):
        self.key = (node_id, alpha, beta)
        self.solution = solution

    def is_hit(self, node_id, alpha, beta):
        return self.key == (node_id, alpha, beta)

    def is_hit_node_only(self, node_id):
        return self.key[0] == node_id

    def __str__(self):
        return f'{self.key[0]}, alpha={self.key[1]}, beta={self.key[2]}, solution={self.solution}'

# for saving the score associated with a set of nodes, given a specific alpha and beta
class S_Transposition:
    """
    for saving the solution for a given set of similar nodes, alpha, and beta
    """

    def __init__(self, s, alpha, beta, solution = (set(), None)):
        self.s = s
        self.limits = (alpha, beta)
        self.solution = solution

    def is_hit(self, node_id, alpha, beta):
        return node_id in self.s and self.limits == (alpha, beta)

    def is_hit_node_only(self, node_id):
        return node_id in self.s

    def __str__(self):
        return f'{self.s}, alpha={self.limits[0]}, beta={self.limits[1]}, solution={self.solution}'

class TranspositionTable:
    def __init__(self):
        self.table = []

    # returns the entry for a specific node, alpha, and beta
    # returns None if not in the table
    def find(self, node_id, alpha, beta):
        for entry in self.table:
            if entry.is_hit(node_id, alpha, beta):
                return entry
        return None

    def find_node_only(self, node_id):
        return [entry for entry in self.table if entry.is_hit_node_only(node_id)]

    # add an entry to the table
    def add(self, item):
        self.table.append(item)

class Stats:
    def __init__(self, max_levels):
        self._nodes_evaluated = [0] * (max_levels + 1)
        self._prunes = [0] * (max_levels + 1)
        self._table_hits = [0] * (max_levels + 1)

    def __str__(self):
        return f'Total nodes evaluated = {sum(self._nodes_evaluated)}' \
            f'\nNodes evaluated per level = { self._nodes_evaluated }' \
            f'\nTotal prunes = {sum(self._prunes)}' \
            f'\nPrunes per level = { self._prunes }' \
            f'\nTable hits per level = { self._table_hits }'

    def node_evaluated(self, level):
        self._nodes_evaluated[level] += 1

    def branch_pruned(self, level):
        self._prunes[level] += 1

    def table_hit(self, level):
        self._table_hits[level] += 1

T = TranspositionTable()

def alpha_beta(node:Node, alpha:float, beta:float, level=0, verbose=False):
    """
    conducts an alpha-beta search

    :param node: current node
    :param alpha:  max has a way to guarantee this score
        if examining plays for min, once he has a way to hold max to this score,
        he can't do better, so there is no need to continue our examination
    :param beta: min has a way to hold max to this score
        if examining plays for max, once he has a way to guarantee this score,
        he can't do better, so there is no need to continue our evaluation
    :param level: starts at 0, incremented as we progress along the tree
    :param verbose: for tracing logic

    :return: the solution: a tuple consisting of the best play and the resulting score
    """

    prefix = '.' * level * 2
    if verbose:
        if node.is_terminal():
            print(f'{prefix}SOLVING {node} (terminal), alpha={alpha}. beta={beta}')
        else:
            print(f"{prefix}SOLVING {node} ({node.player()} to play), alpha={alpha}. beta={beta}")

    global stats
    global T

    # see if we've seen this problem before
    # ignore hit for level 0, since we need a play in addition to a score
    hit = T.find(node.id(), alpha, beta) if level > 0 else None

    # if so, return previously calculated score
    if hit:
        stats.table_hit(level)
        if verbose:
            print(f'{prefix}***** Table hit: {hit} *****')
        return hit.solution

    # if not, evaluate it
    value = node.evaluate()

    # max's play
    if value == node.player_names()[0]:
        solution = (None, float('-inf'))

        #  calculate score for each of max's possible plays
        children = node.get_children().copy()
        while len(children) > 0:
            child = children.pop()
            if verbose:
                print(f'{prefix}Considering {node}->{node.player()} plays {child.get_last_play()}')

            # score is the best max can do by making this play
            _, score = alpha_beta(child, max(solution[1], alpha), beta, level=level+1, verbose=verbose)

            if score > solution[1]:
                if verbose:
                    print(f"{prefix}Result for {node}->{node.player()} plays {child.get_last_play()}: Selecting, since it scores {score}, which is more than current best score of {solution[1]}")
                solution = (child.get_last_play(), score)

                # if this play achieves beta, max can't do better - skip remaining children
                if score >= beta and len(children) > 0:
                    stats.branch_pruned(level + 1)
                    if verbose:
                        print(f"{prefix}Pruning {[child.id() for child in children]}, since {node.opponent()} can always hold {node.player()} to at most {beta} ")
                    break
            else:
                if verbose:
                    print(f"{prefix}Result for {node}->{node.player()} plays {child.get_last_play()}: Rejecting, since it scores {score}, which is not more than current best score is {solution[1]}")

    # min's play
    elif value == node.player_names()[1]:
        solution = (None, float('inf'))

        #  calculate score for each of min's possible plays
        children = node.get_children().copy()
        while len(children) > 0:
            child = children.pop()
            if verbose:
                print(f'{prefix}Considering {node}->{node.player()} plays {child.get_last_play()}')

            # score is the best min can do by making child.play
            _, score = alpha_beta(child, alpha, min(solution[1], beta), level=level+1, verbose=verbose)

            if score < solution[1]:
                if verbose:
                    print(f"{prefix}Result for {node}->{node.player()} plays {child.get_last_play()}: Selecting, since it scores {score}, which is less than current best score of {solution[1]} ")
                solution = (child.get_last_play(), score)

                # if this play achieves alpha, min can't do better - skip remaining children
                if score <= alpha and len(children) > 0:
                    stats.branch_pruned(level + 1)
                    if verbose:
                        print(f"{prefix}Pruning {[child.id() for child in children]}, since {node.opponent()} can always guarantee at least {alpha} ")
                    break
            else:
                if verbose:
                    print(f"{prefix}Result for {node}->{node.player()} plays {child.get_last_play()}: Rejecting, since it scores {score}, which is not less than current best score of {solution[1]} ")

    # if a terminal node, evaluate it
    else:
        solution = (None, value)
        if verbose:
            print(f'{prefix}SOLVED {node}, alpha={alpha}, beta={beta}: {node.show_solution()}')

    # solved
    node.solution = solution
    stats.node_evaluated(level)
    T.add(N_Transposition(node.id(), alpha, beta, solution))
    return solution

def alpha_beta_partitioned(node:Node, alpha:float, beta:float, level=0, verbose=False):
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
    prefix = '.' * level * 2
    if verbose:
        if node.is_terminal():
            print(f'{prefix}SOLVING {node} (terminal), alpha={alpha}. beta={beta}')
        else:
            print(f"{prefix}SOLVING {node} ({node.player()} to play), alpha={alpha}. beta={beta}")

    global stats
    global T

    # see if we've seen this problem before
    # ignore hit for level 0, since we need a play in addition to a score
    hit = T.find(node.id(), alpha, beta) if level > 0 else None

    # if so, return previously calculated solution
    if hit:
        stats.table_hit(level)

        # hit.s will be the problem set - the set of similar problems this node was found to be a member of
        # hit.solution will be a tuple, consisting of the solution set and the score
        # we must calculate the play that will yield some node in the solution set
        best_play = node.get_play(hit.solution[0])
        if verbose:
            print(f'{prefix}***** Table hit: {hit}, play={best_play} *****')

        return hit.s, (best_play, hit.solution[1])

    # if not, evaluate it
    value = node.evaluate()
    all_S = set() # all nodes similar to the nodes we have examined

    # max's play
    if value == node.player_names()[0]:
        solution = (None, float('-inf'))
        solution_id = None
        solution_set = set()

        #  calculate score for each of max's possible plays
        children = node.get_children().copy()
        while len(children) > 0:
            child = children.pop()
            if verbose:
                print(f'{prefix}Considering {node}->{child.get_last_play()}')

            # new_solution_set is the set of nodes similar to this one
            # new_solution is (best_play, score)
            new_solution_set, new_solution = alpha_beta_partitioned(child, max(solution[1], alpha), beta, level=level+1, verbose=verbose)
            all_S |= new_solution_set

            if new_solution[1] > solution[1]:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.get_last_play()}: Selecting, since it scores {new_solution[1]}, which is more than current best score of {solution[1]}")
                solution = (child.get_last_play(), new_solution[1])
                solution_id = child.id()
                solution_set = new_solution_set

                # if this play achieves beta, max can't do better - skip remaining children
                if solution[1] >= beta and len(children) > 0:
                    stats.branch_pruned(level + 1)
                    if verbose:
                        print(f"{prefix}Pruning {[child.id() for child in children]}, since {node.opponent()} can always hold {node.player()} to at most {beta} ")
                    for child in children:
                        all_S = all_S  | child.get_P()
                    break
            else:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.get_last_play()}: Rejecting, since it scores {new_solution[1]}, which is not more than current best score is {solution[1]}")

        # all children evaluated
        node.solution = solution
        if verbose:
            print(f'{prefix}SOLVED {node}: {node.show_solution()}')
            print(f'{prefix}   Our solution set is nodes similar to {solution_id}, namely {solution_set}')
            print(f'{prefix}   Our universe is nodes similar to all solutions we tried {all_S}')
        if solution[1] == MIN_SCORE:
            problem_set = node.get_C(all_S)
            if verbose:
                print(f"{prefix}   Since we can't score from this position, we won't be able to score from any node constrained to the our universe." )
                if len(problem_set) <= 1:
                    print(f'{prefix}   {node} is the only node so constrained')
                else:
                    print(f'{prefix}   These nodes are {problem_set}, which are now defined as similar to {node}')
        else:
            problem_set = node.get_R(solution_set) & node.get_C(all_S)
            if verbose:
                print(f'{prefix}   Nodes that can reach our solution set are {node.get_R(solution_set)}')
                print(f'{prefix}   Nodes that cannot reach outside our universe are {node.get_C(all_S)}')
                if len(problem_set) <= 1:
                    print(f'{prefix}   {node} is the only node so constrained')
                else:
                    print(f'{prefix}   These nodes are {problem_set}, which are now defined as similar to {node}')

    # min's play
    elif value == node.player_names()[1]:
        solution = (None,  float('inf'))
        solution_id = None
        solution_set = set()

        #  calculate score for each of min's possible plays
        children = node.get_children().copy()
        while len(children) > 0:
            child = children.pop()
            if verbose:
                print(f'{prefix}Considering {node}->{child.get_last_play()}')

            # new_solution_set is the set of nodes similar to this one
            # new_solution is (best_play, score)
            new_solution_set, new_solution = alpha_beta_partitioned(child, alpha, min(solution[1], beta), level=level+1, verbose=verbose)
            all_S |= new_solution_set

            if new_solution[1] < solution[1]:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.get_last_play()}: Selecting, since it scores {new_solution[1]}, which is less than current best score of {solution[1]} ")
                solution = (child.get_last_play(), new_solution[1])
                solution_id = child.id()
                solution_set = new_solution_set

                # if this play achieves alpha, min can't do better - skip remaining children
                if solution[1] <= alpha and len(children) > 0:
                    stats.branch_pruned(level + 1)
                    if verbose:
                        print(f"{prefix}Pruning {[child.id() for child in children]}, since {node.opponent()} can always guarantee at least {alpha} ")
                    for child in children:
                        all_S = all_S  | child.get_P()
                    break
            else:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.get_last_play()}: Rejecting, since it scores {new_solution[1]}, which is not less than current best score of {solution[1]} ")

        # all children evaluated
        node.solution = solution
        if verbose:
            print(f'{prefix}SOLVED {node}: {node.show_solution()}')
            print(f'{prefix}   Our solution set is nodes similar to {solution_id}, namely {solution_set}')
            print(f'{prefix}   Our universe is nodes similar to all solutions we tried {all_S}')
        if solution[1] == MAX_SCORE:
            problem_set = node.get_C(all_S)
            if verbose:
                print(f"{prefix}   Since we must fail from this node, we must fail from any node constrained to our universe." )
                if len(problem_set) <= 1:
                    print(f'{prefix}   {node} is the only node so constrained')
                else:
                    print(f'{prefix}   These nodes are {problem_set}, which are now defined as similar to {node}')
        else:
            problem_set = node.get_R(solution_set) & node.get_C(all_S)
            if verbose:
                print(f'{prefix}   Nodes that can reach our solution set are {node.get_R(solution_set)}')
                print(f'{prefix}   Nodes that cannot reach outside our universe are {node.get_C(all_S)}')
                if len(problem_set) <= 1:
                    print(f'{prefix}   {node} is the only intersection of these two sets - hence no nodes are similar to {node}')
                else:
                    print(f'{prefix}   The intersection of these two sets ({problem_set}) is now defined as similar to {node}')

    # if a terminal node, evaluate it
    # return score and a set of nodes that evaluate to the same value
    else:
        solution = (None, value)
        problem_set = node.get_P()
        solution_set = set()

        if verbose:
            print(f'{prefix}SOLVED {node}, alpha={alpha}, beta={beta}: {node.show_solution()}')
            if len(problem_set) <= 1:
                print(f'{prefix}   No nodes are similar to {node}')
            else:
                print(f'{prefix}   Terminal nodes that have the same value ({problem_set}) are now defined as similar to {node}')

    # solved
    if node.id() not in problem_set:
        raise Exception(f'{prefix}Problem set does not contain the specified problem node')
    stats.node_evaluated(level)
    T.add(S_Transposition(problem_set, alpha, beta, (solution_set, solution[1])))
    if verbose:
        print(f"{prefix}Added to T: {T.table[-1]}")
    return problem_set, solution

def show_T_entry(node_id):
    print(f'Entries for {node_id}:')
    for entry in T.find_node_only(node_id):
        print(f'   {entry}')

def play(node, alpha, beta, verbose=False):
    if not node.is_terminal():
        solution = alpha_beta(node, alpha, beta, verbose=verbose)
        return node.get_child(play=solution[0])
    return None

stats = Stats(6)
node = TestNode(max_plays = MAX_PLAYS)

alpha_beta_partitioned(node, 0, MAX_PLAYS, verbose=True)
print(f'Final solution = {node.show_solution()}')
print(stats)
print(f'Table size = {len(T.table)}')

# alpha = 0
# beta = MAX_PLAYS
# while node:
#     next_node = play(node, alpha, beta, verbose=False)
#     print(f'Solution = {node.show_solution()}')
#     print(stats)
#     stats = Stats(MAX_PLAYS)
#     node = next_node


# for entry in T.table:
#     print(entry)

# node = TestNode()
# node.plays = 'AAB'
# print(node.get_P())
# print(node.get_R(node.get_P()))
# print(node.get_C({'AAB', 'AAA', 'ABA'}))






