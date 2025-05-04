from collections import Counter
from pyexpat import ExpatError

MAX_PLAYS = 6
MIN_SCORE = 0.0
MAX_SCORE = float(MAX_PLAYS)

class TestNode:
    def __init__(self, play=None, parent=None):
        self.plays = parent.plays if parent else ''
        if play:
            self.plays += play
        self.solution = (None, None)

    def id(self):
        return self.plays

    def current_play(self):
        if len(self.plays) == 0:
            return None
        else:
            return self.plays[-1]

    def get_children(self):
        if self.is_terminal():
            return []
        else:
            return [TestNode('B', self), TestNode('A', self)]

    # returns score for a terminal node
    # returns next to play ('min' or 'max') for a non-terminal node
    def evaluate(self):
        if self.is_terminal():
            score = 0
            for play in self.plays:
                if play == 'A':
                    score += 1
            return score
        else:
            return self.next_to_play()

    # if this is a terminal node, return a set containing the ids of terminal nodes with the same value
    # otherwise return a set containing only this id
    def P(self):
        if not self.is_terminal():
            return { self.id() }
        else:
            return TestNode._permutations(*self._counts())

    # S is a set of node ids
    # return the ids of nodes that have an immediate successor in set S
    def R(self, S):
        return { id[:-1] for id in S }

    # S is a set of node ids
    # return the ids of nodes for which all immediate successors are in set S
    def C(self, S):
        counts = Counter([id[:-1] for id in S])
        return {id for id, count in counts.items() if count == 2}

    def show_solution(self):
        if self.is_terminal():
            return f'Terminal node, score={self.evaluate()}'

        if not self.solution[0]:
            return 'Not solved'

        if self.next_to_play() == 'max':
            return  f'Max can play {self.solution[0]} to guarantee a score of {self.solution[1]}'
        else:
            return f'Min can play {self.solution[0]} to hold max to a score of {self.solution[1]}'

    # private helper functions
    def is_terminal(self):
        return len(self.plays) >= MAX_PLAYS

    def next_to_play(self):
        return 'max' if len(self.plays) % 2 == 0 else 'min'

    def _counts(self):
        counts = [0, 0]
        for card in self.plays:
            if card == 'A':
                counts[0] += 1
            else:
                counts[1] += 1
        return tuple(counts)

    @staticmethod
    def _permutations(a_num, b_num):
        if a_num == 0 and b_num == 0:
            return {}
        if a_num == 0:
            return {'B' * b_num}
        if b_num == 0:
            return {'A' * a_num}
        r = ['A' + item for item in TestNode._permutations(a_num - 1, b_num)]
        r.extend(['B' + item for item in TestNode._permutations(a_num, b_num - 1)])
        return set(r)

    def __str__(self):
        return self.plays if len(self.plays) > 0 else 'root'

# for saving the score associated with a node, given a specific alpha and beta
class N_Transposition:
    def __init__(self, node_id, alpha, beta, score=None):
        self.key = (node_id, alpha, beta)
        self.score = score

    def is_hit(self, node_id, alpha, beta):
        return self.key == (node_id, alpha, beta)

    def is_hit_node_only(self, node_id):
        return self.key[0] == node_id

    def __str__(self):
        return f'{self.key[0]}, alpha={self.key[1]}, beta={self.key[2]}, score={self.score}'

# for saving the score associated with a set of nodes, given a specific alpha and beta
class S_Transposition:
    def __init__(self, s, alpha, beta, score=None):
        self.s = s
        self.limits = (alpha, beta)
        self.score = score

    def is_hit(self, node_id, alpha, beta):
        return node_id in self.s and self.limits == (alpha, beta)

    def is_hit_node_only(self, node_id):
        return node_id in self.s

    def __str__(self):
        return f'{self.s}, alpha={self.limits[0]}, beta={self.limits[1]}, score={self.score}'

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

stats = Stats(MAX_PLAYS)
T = TranspositionTable()

"""
alpha - max has a way to guarantee this score 
    if examining plays for min, once he has a way to hold max to this score,
    he can't do better, so there is no need to continue our examination 
beta - min has a way to hold max to this score
    if examining plays for max, once he has a way to guarantee this score,
    he can't do better, so there is no need to continue our evaluation 
    
returns the score for the specified node
"""

def alpha_beta(node:TestNode, alpha:float, beta:float, level=0, verbose=False):
    prefix = '.' * level * 2
    if verbose:
        if node.is_terminal():
            print(f'{prefix}SOLVING {node} (terminal), alpha={alpha}. beta={beta}')
        else:
            print(f"{prefix}SOLVING {node} ({node.next_to_play()} to play), alpha={alpha}. beta={beta}")

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
        return hit.score

    # if not, evaluate it
    value = node.evaluate()
    best_play = None

    # max's play
    if value == 'max':
        best_score = MIN_SCORE
        best_child_id = None

        #  calculate score for each of max's possible plays
        children = node.get_children().copy()
        while len(children) > 0:
            child = children.pop()
            if verbose:
                print(f'{prefix}Considering {node}->{child.current_play()}')

            # score is the best max can do by making child.play
            score = alpha_beta(child, max(best_score, alpha), beta, level=level+1, verbose=verbose)

            if score > best_score:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.current_play()}: Selecting for now, since it scores {score}, which is more than current best score of {best_score}")
                best_play = child.current_play()
                best_score = score
                best_child_id = child.id()

                # if this play achieves beta, max can't do better - skip remaining children
                if score >= beta and len(children) > 0:
                    stats.branch_pruned(level + 1)
                    if verbose:
                        print(f"{prefix}Pruning {[child.id() for child in children]}, since min can always hold max to at most {beta} ")
                    break
            else:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.current_play()}: Rejecting, since it scores {score}, which is not more than current best score is {best_score}")

        # all children evaluated
        node.solution = (best_play, best_score)

    # min's play
    elif value == 'min':
        best_score = MAX_SCORE
        best_child_id = None

        #  calculate score for each of min's possible plays
        children = node.get_children().copy()
        while len(children) > 0:
            child = children.pop()
            if verbose:
                print(f'{prefix}Considering {node}->{child.current_play()}')

            # score is the best min can do by making child.play
            score = alpha_beta(child, alpha, min(best_score, beta), level=level+1, verbose=verbose)

            if score < best_score:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.current_play()}: Selecting for now, since it scores {score}, which is less than current best score of {best_score} ")
                best_play = child.current_play()
                best_score = score
                best_child_id = child.id()

                # if this play achieves alpha, min can't do better - skip remaining children
                if score <= alpha and len(children) > 0:
                    stats.branch_pruned(level + 1)
                    if verbose:
                        print(f"{prefix}Pruning {[child.id() for child in children]}, since max can always guarantee at least {alpha} ")
                    break
            else:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.current_play()}: Rejecting, since it scores {score}, which is not less than current best score of {best_score} ")

        # all children evaluated
        node.solution = (best_play, best_score)

    # if a terminal node, evaluate it
    else:
        best_score = value

        node.solution = (best_play, best_score)
        if verbose:
            print(f'{prefix}SOLVED {node}, alpha={alpha}, beta={beta}: {node.show_solution()}')

    # solved
    stats.node_evaluated(level)
    T.add(N_Transposition(node.id(), alpha, beta, best_score))
    return float(best_score)
"""
alpha - max has a way to guarantee this score 
    if examining plays for min, once he has a way to hold max to this score,
    he can't do better, so there is no need to continue our examination 
beta - min has a way to hold max to this score
    if examining plays for max, once he has a way to guarantee this score,
    he can't do better, so there is no need to continue our evaluation 
    
returns a score and a set S of nodes that have the same score
S will include the 'node' passed as a parameter and may include other nodes as well, which we call 'similar nodes'
"""
def alpha_beta_partitioned(node:TestNode, alpha:float, beta:float, level=0, verbose=False):
    prefix = '.' * level * 2
    if verbose:
        if node.is_terminal():
            print(f'{prefix}SOLVING {node} (terminal), alpha={alpha}. beta={beta}')
        else:
            print(f"{prefix}SOLVING {node} ({node.next_to_play()} to play), alpha={alpha}. beta={beta}")

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
        return hit.score, hit.s

    # if not, evaluate it
    value = node.evaluate()
    best_play = None
    all_S = set() # all nodes similar to the nodes we have examined

    # max's play
    if value == 'max':
        best_score = MIN_SCORE
        best_child_id = None
        best_child_S = set()

        #  calculate score for each of max's possible plays
        children = node.get_children().copy()
        while len(children) > 0:
            child = children.pop()
            if verbose:
                print(f'{prefix}Considering {node}->{child.current_play()}')

            # score is the best max can do by making child.play
            # S is the set of similar nodes (nodes that have this score)
            score, S = alpha_beta_partitioned(child, max(best_score, alpha), beta, level=level+1, verbose=verbose)
            all_S |= S

            if score > best_score:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.current_play()}: Selecting for now, since it scores {score}, which is more than current best score of {best_score}")
                best_play = child.current_play()
                best_score = score
                best_child_id = child.id()
                best_child_S = S

                # if this play achieves beta, max can't do better - skip remaining children
                if score >= beta and len(children) > 0:
                    stats.branch_pruned(level + 1)
                    if verbose:
                        print(f"{prefix}Pruning {[child.id() for child in children]}, since min can always hold max to at most {beta} ")
                    for child in children:
                        all_S = all_S  | child.P()
                    break
            else:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.current_play()}: Rejecting, since it scores {score}, which is not more than current best score is {best_score}")

        # all children evaluated
        node.solution = (best_play, best_score)
        if verbose:
            print(f'{prefix}SOLVED {node}: {node.show_solution()}')
            print(f'{prefix}   Our solution set is nodes similar to {best_child_id}, namely {best_child_S}')
            print(f'{prefix}   Our universe is nodes similar to all solutions we tried {all_S}')
        if best_score == MIN_SCORE:
            best_S = node.C(all_S)
            if verbose:
                print(f"{prefix}   Since we can't score from this position, we won't be able to score from any node constrained to the our universe." )
                if len(best_S) <= 1:
                    print(f'{prefix}   {node} is the only node so constrained')
                else:
                    print(f'{prefix}   These nodes are {best_S}, which are now defined as similar to {node}')
        else:
            best_S = node.R(best_child_S) & node.C(all_S)
            if verbose:
                print(f'{prefix}   Nodes that can reach our solution set are {node.R(best_child_S)}')
                print(f'{prefix}   Nodes that cannot reach outside our universe are {node.C(all_S)}')
                if len(best_S) <= 1:
                    print(f'{prefix}   {node} is the only node so constrained')
                else:
                    print(f'{prefix}   These nodes are {best_S}, which are now defined as similar to {node}')

    # min's play
    elif value == 'min':
        best_score = MAX_SCORE
        best_child_id = None
        best_child_S = set()

        #  calculate score for each of min's possible plays
        children = node.get_children().copy()
        while len(children) > 0:
            child = children.pop()
            if verbose:
                print(f'{prefix}Considering {node}->{child.current_play()}')

            # score is the best min can do by making child.play
            # S is the set of similar nodes (nodes that have this score)
            score, S = alpha_beta_partitioned(child, alpha, min(best_score, beta), level=level+1, verbose=verbose)
            all_S |= S

            if score < best_score:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.current_play()}: Selecting for now, since it scores {score}, which is less than current best score of {best_score} ")
                best_play = child.current_play()
                best_score = score
                best_child_id = child.id()
                best_child_S = S

                # if this play achieves alpha, min can't do better - skip remaining children
                if score <= alpha and len(children) > 0:
                    stats.branch_pruned(level + 1)
                    if verbose:
                        print(f"{prefix}Pruning {[child.id() for child in children]}, since max can always guarantee at least {alpha} ")
                    for child in children:
                        all_S = all_S  | child.P()
                    break
            else:
                if verbose:
                    print(f"{prefix}Result for {node}->{child.current_play()}: Rejecting, since it scores {score}, which is not less than current best score of {best_score} ")

        # all children evaluated
        node.solution = (best_play, best_score)
        if verbose:
            print(f'{prefix}SOLVED {node}: {node.show_solution()}')
            print(f'{prefix}   Our solution set is nodes similar to {best_child_id}, namely {best_child_S}')
            print(f'{prefix}   Our universe is nodes similar to all solutions we tried {all_S}')
        if best_score == MAX_SCORE:
            best_S = node.C(all_S)
            if verbose:
                print(f"{prefix}   Since we must fail from this node, we must fail from any node constrained to our universe." )
                if len(best_S) <= 1:
                    print(f'{prefix}   {node} is the only node so constrained')
                else:
                    print(f'{prefix}   These nodes are {best_S}, which are now defined as similar to {node}')
        else:
            best_S = node.R(best_child_S) & node.C(all_S)
            if verbose:
                print(f'{prefix}   Nodes that can reach our solution set are {node.R(best_child_S)}')
                print(f'{prefix}   Nodes that cannot reach outside our universe are {node.C(all_S)}')
                if len(best_S) <= 1:
                    print(f'{prefix}   {node} is the only intersection of these two sets - hence no nodes are similar to {node}')
                else:
                    print(f'   The intersection of these two sets ({best_S}) are now defined as similar to {node}')

    # if a terminal node, evaluate it
    # return score and a set of nodes that evaluate to the same value
    else:
        best_score = value
        best_S = node.P()

        node.solution = (best_play, best_score)
        if verbose:
            print(f'{prefix}SOLVED {node}, alpha={alpha}, beta={beta}: {node.show_solution()}')
            if len(best_S) <= 1:
                print(f'{prefix}   No nodes are similar to {node}')
            else:
                print(f'{prefix}   Terminal nodes that have the same value ({best_S}) are now defined as similar to {node}')

    # solved
    if len(best_S) == 0:
        raise Exception(f'{prefix}Empty best_S')
    stats.node_evaluated(level)
    T.add(S_Transposition(best_S, alpha, beta, best_score))
    return float(best_score), best_S

def show_T_entry(node_id):
    print(f'Entries for {node_id}:')
    for entry in T.find_node_only(node_id):
        print(f'   {entry}')

def play(node, alpha, beta, verbose=False):
    if not node.is_terminal():
        alpha_beta_partitioned(node, alpha, beta, verbose=verbose)
        for child in node.get_children():
            if child.current_play() == node.solution[0]:
                return child
    return None


node = TestNode()
# alpha_beta_partitioned(node, 0, MAX_PLAYS, verbose=True)
# print(f'Final solution = {node.show_solution()}')
# print(stats)
# print(f'Table size = {len(T.table)}')

alpha = 0
beta = MAX_PLAYS
while node:
    next_node = play(node, alpha, beta, verbose=False)
    print(f'Solution = {node.show_solution()}')
    print(stats)
    stats = Stats(MAX_PLAYS)
    node = next_node


# for entry in T.table:
#     print(entry)

# node = TestNode()
# node.plays = 'AAB'
# print(node.P())
# print(node.R(node.P()))
# print(node.C({'AAB', 'AAA', 'ABA'}))






