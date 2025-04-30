class TestNode:
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

    def label(self):
        str = ''
        for n in range(len(self.min_plays)):
            str += self.max_plays[n] + self.min_plays[n]
        if len(self.max_plays) > len(self.min_plays):
            str += self.max_plays[-1]
        return str if len(str) > 0 else 'root'

    # for purposes of determining if an equivalent node is in the Transition Table
    # for TestNode, two nodes are equivalent if they have the same number of A's and B's played,
    #   regardless of which player played them
    def id(self):
        count = [0, 0]
        for play in self.min_plays + self.max_plays:
            if play == 'A':
                count[0] += 1
            else:
                count[1] += 1
        return tuple(count)

    def get_children(self):
        return [TestNode('A', self), TestNode('B', self)]

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

    # private helper functions
    def _is_terminal(self):
        return len(self.min_plays) + len(self.max_plays) >= 6

    def _next_to_play(self):
        return 'max' if len(self.max_plays) <= len(self.min_plays) else 'min'

    def __str__(self):
        if self._is_terminal():
            return self.label() + f': Terminal node, score={self.evaluate()}'

        if not self.solution[0]:
            return self.label()

        if self._next_to_play() == 'max':
            return self.label() + f': Max can play {self.solution[0]} to guarantee a score of {self.solution[1]}'
        else:
            return self.label() + f': Min can play {self.solution[0]} to hold max to a score of {self.solution[1]}'

# for saving the score associated with a node, given a specific alpha and beta
class Transposition:
    # node's __repr__ method must return a tuple
    def __init__(self, node, alpha, beta, score=None):
        self.key = node.id() + (alpha, beta)
        self.score = score

    def __str__(self):
        return f'{self.key[:-2]}, alpha={self.key[-2]}, beta={self.key[-1]}, score={self.score}'

class TranspositionTable:
    def __init__(self):
        self.table = []

    # returns the entry for a specific node, alpha, and beta
    # returns None if not in the table
    def find(self, node, alpha, beta):
        new_entry = Transposition(node, alpha, beta)
        for entry in self.table:
            if entry.key == new_entry.key:
                return entry
        return None

    # adds an entry to the table if not already there
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
def alpha_beta(node:TestNode, alpha:float, beta:float, verbose=False):
    if verbose:
        print(f'Solving {node.label()}, alpha={alpha}. beta={beta}')

    global branches_traversed
    global T
    global table_hits

    # see if we've seen this problem before
    hit = T.find(node, alpha, beta)

    # if so, return previously calculated score
    if hit:
        if verbose:
            print(f'***** Table hit: {hit} *****')
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
            if verbose:
                print(f'Solved child {child.label()}: score={score}')

            # min can hold max to beta
            # if this play achieves beta, max can't do better - return this solution
            if score >= beta:
                best_play = child.play
                best_score = score
                if verbose:
                    print(f"   {node.label()}->Pruning max's remaining choices ")
                break

            if score > best_score:
                best_play = child.play
                best_score = score
                if verbose:
                    print(f"   {node.label()}->Updating min's best choice to {child.play}")
            else:
                if verbose:
                    print(f'   {node.label()}->Rejecting choice {child.play}')

    # min's play
    elif eval == 'min':
        best_score = float('inf')

        #  calculate score for each of min's possible plays
        for child in node.get_children():
            # score is the best min can do by making child.play
            score = alpha_beta(child, alpha, min(best_score, beta), verbose)
            if verbose:
                print(f'Solved child {child.label()}: score={score}')

            # max can guarantee alpha
            # if this play holds max to alpha, min can't do better - return this solution
            if score <= alpha:
                best_play = child.play
                best_score = score
                if verbose:
                    print(f"   {node.label()}->Pruning min's remaining choices ")
                break

            if score < best_score:
                best_play = child.play
                best_score = score
                if verbose:
                    print(f"   {node.label()}->Updating min's best choice to {child.play}")
            else:
                if verbose:
                    print(f'   {node.label()}->Rejecting choice {child.play}')

    # if a terminal node, evaluate it and return score
    else:
        branches_traversed += 1
        best_score = eval

    # solved
    node.solution = (best_play, best_score)
    T.add(node, alpha, beta, best_score)
    if verbose:
        print(f'Solved node: {node}')
        print(f'Adding to Transposition Table: {T.table[-1]}')
    return best_score

node = TestNode()
alpha_beta(node, float('-inf'), float('inf'), True)
print(f'Final solution = {node}')
print(f"Branches traversed = {branches_traversed}")
print(f'Table hits = {table_hits}')

print(f'\nTransposition Table:')
for item in T.table:
    print(item)


