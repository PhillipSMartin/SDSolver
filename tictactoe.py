from alpha_beta import *

class TicTacToeNode(Node):
    def __init__(self, parent=None, play=None, next_to_play=None):
        super().__init__(parent=parent, play=play, next_to_play=next_to_play, default_state=['.'] * 9)

    def id(self):
        return '|'.join( [''.join(self.state[n:n+3]) for n in range(0,9,3)] )

    @staticmethod
    def player_names():
        """ returns list of players' names """
        return ['X', 'O']

    def compute_state(self, play):
        """
        calculates new state based on current state and specified play
        default implementation is to create a string from the list of plays

        :param play: the play that produces this node from the parent
        :return: new state
        """
        self.state[play] = self.opponent()
        return self.state

    def compute_score(self) -> float or None:
        """
        calculates the score of a terminal node
        must be overridden

        :return: score for a terminal node; None for a non-terminal node
        """
        if self.is_terminal():
            if self._winner() == 'X':
                return 1
            elif self._winner() == 'O':
                return 0
            else:
                return 0.5
        return None

    def get_options(self):
        """
        returns a list of possible plays
        must be overridden

        nodes should be returned in opposite order of the order you want to evaluate them in
        """
        return [index for index, play in enumerate(self.state) if play == '.']

    def get_child(self, play):
        """
        returns a child of this node resulting from making the specified play
        must be overridden
        """
        return TicTacToeNode(parent=self, play=play)

    def get_state(self):
        return self.state.copy()

    def show(self):
        s = ' ' + ' | '.join(self.state[0:3]) + '\n-----------\n '
        s += ' | '.join(self.state[3:6]) + '\n-----------\n '
        s += ' | '.join(self.state[6:9]) + '\n'
        return s

    def _winner(self):
        for n in range(0,9,3):
            x = set(self.state[n:n+3])
            if len(x) == 1 and '.' not in x:
                return list(x)[0]
        for n in range(0, 3):
            x = set(self.state[n:n+9:3])
            if len(x) == 1 and '.' not in x:
                return list(x)[0]
        x = set(self.state[0:12:4])
        if len(x) == 1 and '.' not in x:
            return list(x)[0]
        x = set(self.state[2:8:2])
        if len(x) == 1 and '.' not in x:
            return list(x)[0]
        return None

    def is_terminal(self):
        return '.' not in self.state or self._winner()


node = TicTacToeNode()
game = Game(min_score = 0.0, max_score = 1.0, max_levels = 9)
game.play(node,0, 6, partitioned=True, verbose=False)
game.show_stats()