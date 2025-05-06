import alpha_beta as ab

class TicTacToeNode:
    def __init__(self, parent=None, player=None, play=None):
        self.boxes = parent.boxes.copy() if parent else ['.'] * 9
        if play is not None:
            self.current_play = play
            self.boxes[play] = player
        else:
            self.current_play = None

    def id(self):
        return '|'.join( [''.join(self.boxes[n:n+3]) for n in range(0,9,3)] )

    def __str__(self):
        return self.id()

    # returns score for a terminal node
    # returns next to play ('min' or 'max') for a non-terminal node
    def evaluate(self):
        if self.is_terminal():
            if self._winner() == 'X':
                return 1
            elif self._winner() == 'O':
                return -1
            else:
                return 0
        return None

    def show(self):
        s = ' ' + ' | '.join(self.boxes[0:3]) + '\n-----------\n '
        s += ' | '.join(self.boxes[3:6]) + '\n-----------\n '
        s += ' | '.join(self.boxes[6:9]) + '\n'
        return s

    def _winner(self):
        for n in range(0,9,3):
            x = set(self.boxes[n:n+3])
            if len(x) == 1 and '.' not in x:
                return list(x)[0]
        for n in range(0, 3):
            x = set(self.boxes[n:n+9:3])
            if len(x) == 1 and ' ' not in x:
                return list(x)[0]
        x = set(self.boxes[0:12:4])
        if len(x) == 1 and '.' not in x:
            return list(x)[0]
        x = set(self.boxes[2:8:2])
        if len(x) == 1 and '.' not in x:
            return list(x)[0]
        return None

    def is_terminal(self):
        return '.' not in self.boxes or self._winner()

    def next_to_play(self):
        if self.current_play is None or self.boxes[self.current_play] == 'O':
            return 'X'
        else:
            return 'O'

    def get_children(self):
        if self.is_terminal():
            return []
        else:
            return [TicTacToeNode(parent=self, player=self.next_to_play(), play=index) for index, player in enumerate(self.boxes) if player == '.']

stats = ab.Stats(9)

node = TicTacToeNode()
ab.alpha_beta(node, -1, 1, verbose=True)
print(f'Final solution = {node.show_solution()}')
print(stats)
print(f'Table size = {len(T.table)}')

