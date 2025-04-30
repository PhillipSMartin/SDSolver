class Node:
    def __init__(self, play=None, node=None):
        self.max_plays = node.max_plays.copy() if node else []
        self.min_plays = node.min_plays.copy() if node else []
        self.solution = (None, None)
        self.play = play
        if play:
            if self._next_to_play() == 'max':
                self.max_plays.append(play)
            else:
                self.min_plays.append(play)

    def get_children(self):
        return [Node('A', self), Node('B', self)]

    # returns score for a terminal node
    # returns next to play ('min' or 'max') for a non-terminal node
    def evaluate(self):
        if self._is_terminal():
            score = 0
            for play in self.min_plays + self.max_plays:
                if play == 'A':
                    score += 1
            return score
        else:
            return self._next_to_play()

    # defines equality for purposes of determining if the node is in the Cache
    # for this Node, two nodes are equal if they have the same number of A's and B's played,
    #   regardless of which player played them
    def equals(self, node):
        return self._count_by_play == node._count_by_play

    # private helper functions
    def _is_terminal(self):
        return len(self.min_plays) + len(self.max_plays) >= 4

    def _next_to_play(self):
        return 'max' if len(self.max_plays) <= len(self.min_plays) else 'min'

    def _count_by_play(self):
        count = [0, 0]
        for play in self.min_plays + self.max_plays:
            if play == 'A':
                count[0] += 1
            else:
                count[1] += 1
        return count

    def __str__(self):
        str = ''
        for n in range(len(self.min_plays)):
            str += self.max_plays[n] + self.min_plays[n]
        if len(self.max_plays) > len(self.min_plays):
            str += self.max_plays[-1]

        if self._is_terminal():
            return str + f': Terminal node, score={self.evaluate()}'

        if not self.solution[0]:
            return str

        if self._next_to_play() == 'max':
            return str + f': Max can play {self.solution[0]} to guarantee a score of {self.solution[1]}'
        else:
            return str + f': Min can play {self.solution[0]} to hold max to a score of {self.solution[1]}'

# for saving the score associated with a node, given a specific alpha and beta
class Transposition:
    def __init__(self, node, alpha, beta, score=None):
        self.node = node
        self.alpha = alpha
        self.beta = beta
        self.score = score

    def equals(self, node, alpha, beta):
        return self.node.equals(node) and self.alpha == alpha and self.beta == beta

    def __str__(self):
        return f'{self.node}, alpha={self.alpha}, beta={self.beta}, score={self.score}'

class TranspositionTable:
    def __init__(self):
        self.table = []

    # returns the entry for a specific node, alpha, and beta
    # returns None if not in the table
    def find(self, node, alpha, beta):
        for entry in self.table:
            if entry.equals(node, alpha, beta):
                return entry
        return None

    # adds an entry to the cache if not already there
    def add(self, node, alpha, beta, score):
        if not self.find(node, alpha, beta):
            self.table.append(Transposition(node, alpha, beta, score))

branches_traversed = 0
table_hits = 0
T = TranspositionTable()

"""
alpha - max has a way to guarantee this score 
    if evaluating an alternative play for max, once min has a way to hold max to this score,
    there is no need to continue our evaluation 
beta - min has a way to hold max to this score
    if evaluating an alternative play for min, once max has a way to guarantee this score,
    there is no need to continue our evaluation 
"""
def alpha_beta(node:Node, alpha:float, beta:float, verbose=False):
    if verbose:
        print(f'Solving {node}, alpha={alpha}. beta={beta}')

    global branches_traversed
    global T
    global table_hits

    # see if we've seen this problem before
    hit = T.find(node, alpha, beta)

    # if so, return previously calculated score
    if hit:
        if verbose:
            print(f'Table hit: {hit}')
        table_hits += 1
        return hit.score

    # if not, evaluate it
    eval = node.evaluate()
    best_play = None

    # max's play
    if eval == 'max':
        best_score = float('-inf')

        #  calculate score for each of max's possible plays
        for child in node.get_children():
            # score is the best max can do by making child.play
            score = alpha_beta(child, max(best_score, alpha), beta, verbose)

            # min can hold max to beta
            # if this play achieves beta, max can't do better - return this solution
            if score >= beta:
                node.solution = (child.play, score)
                T.add(node, alpha, beta, best_score)
                if verbose:
                    print(f"Pruning max's remaining choices - {node} and min has a way to hold max to {beta}")
                    print(f'Adding to Transposition Table: {T.table[-1]}')
                return score

            if score > best_score:
                if verbose:
                    print('This is best choice so far')
                best_play = child.play
                best_score = score
            else:
                if verbose:
                    print('This choice is rejected')

    # min's play
    elif eval == 'min':
        best_score = float('inf')

        #  calculate score for each of min's possible plays
        for child in node.get_children():
            # score is the best min can do by making child.play
            score = alpha_beta(child, alpha, min(best_score, beta), verbose)

            # max can guarantee alpha
            # if this play holds max to alpha, min can't do better - return this solution
            if score <= alpha:
                node.solution = (child.play, score)
                T.add(node, alpha, beta, score)
                if verbose:
                    print(f"Pruning min's remaining choices - {node} and max has a way to guarantee {alpha}")
                    print(f'Adding to Transposition Table: {T.table[-1]}')
                return score

            if score < best_score:
                if verbose:
                    print('This is best choice so far')
                best_play = child.play
                best_score = score
            else:
                if verbose:
                    print('This choice is rejected')

    # if a terminal node, evaluate it and return score
    else:
        branches_traversed += 1
        best_score = eval

    # solved
    node.solution = (best_play, best_score)
    T.add(node, alpha, beta, best_score)
    if verbose:
        print(f'Solved: {node}')
        print(f'Adding to Transposition Table: {T.table[-1]}')
    return best_score

node = Node()
alpha_beta(node, float('-inf'), float('inf'), True)
print(f'Final solution{node}')
print(f"Branches traversed = {branches_traversed}")
print(f'Table hits = {table_hits}')

print(f'\nTransposition Table:')
for item in T.table:
    print(item)


