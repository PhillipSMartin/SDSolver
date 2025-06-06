from alpha_beta import *
from ruleset import IdSet

class TicTacToeNode(Node):
    x_wins:set = None
    o_wins:set = None
    ties:set = None

    def __init__(self, parent=None, play=None, next_to_play=None):
        super().__init__(parent=parent, play=play, next_to_play=next_to_play, default_state=['.'] * 9)

    def id(self):
        return self.format_id(self.state)

    @staticmethod
    def format_id(_state):
        return '|'.join( [''.join(_state[n:n+3]) for n in range(0,9,3)] )

    @staticmethod
    def from_id(_id:str):
        node = TicTacToeNode()
        node.state = list(_id.replace('|',''))
        return node

    @staticmethod
    def player_names():
        """ returns list of players' names """
        return ['X', 'O']

    @staticmethod
    def generate_ids(n:int=9):
        if n == 1:
            return ['.', 'X', 'O']
        else:
            states = [p + x for p in ['.', 'X', 'O'] for x in TicTacToeNode.generate_ids(n - 1)]
        if n < 9:
            return states
        else:
            return [TicTacToeNode.format_id(s) for s in states if s.count("X") in {s.count("O"), s.count("O") + 1}]

    @classmethod
    def initialize_partitions(cls):
        all_states = TicTacToeNode.generate_ids()
        cls.x_wins = set([x for x in all_states if TicTacToeNode.from_id(x)._winner() == 'X' and x.count('X') == 1 + x.count('O')])
        cls.o_wins = set([x for x in all_states if TicTacToeNode.from_id(x)._winner() == 'O'and x.count('X') == x.count('O')])
        cls.ties = set([x for x in all_states if TicTacToeNode.from_id(x)._winner() is None])

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
        return TicTacToeNode(parent=self, play=int(play))

    def get_state(self):
        return self.state.copy()

    def show(self):
        s = ' ' + ' | '.join(self.state[0:3]) + '\n-----------\n '
        s += ' | '.join(self.state[3:6]) + '\n-----------\n '
        s += ' | '.join(self.state[6:9]) + '\n'
        return s

    def is_similar(self) -> IdSet:
        """
        if this is a terminal node, returns a set containing the ids of terminal nodes with the same value
        otherwise returns a set containing only this id
        """
        if not self.is_terminal():
            return IdSet({ self.id() }, f"nodes similar to {self}")
        else:
            winner = self._winner()
            if winner is None:
                return IdSet( TicTacToeNode.ties, "nodes ending in a tie")
            else:
                return IdSet( TicTacToeNode.x_wins if winner == 'X' else TicTacToeNode.o_wins, f"nodes where {winner} wins")
    #
    # def can_precede(self, id_set:IdSet, descr:str=None)->IdSet:
    #     return IdSet({ node_id[:-1] for node_id in id_set.s }, descr)
    #
    # def can_precede_only(self, id_set:IdSet, descr:str=None) -> IdSet:
    #     counts = Counter([node_id[:-1] for node_id in id_set.s])
    #     return IdSet({node_id for node_id, count in counts.items() if count == 2}, descr)

    def _is_winner(self, player):
        for n in range(0, 9, 3):
            x = set(self.state[n:n+3])
            if len(x) == 1 and player in x:
                return True
        for n in range(0, 3):
            x = set(self.state[n:n+9:3])
            if len(x) == 1 and player in x:
                return True
        x = set(self.state[0:12:4])
        if len(x) == 1 and player in x:
                return True
        x = set(self.state[2:8:2])
        if len(x) == 1 and player in x:
                return True
        return False

    def _winner(self):
        if self._is_winner('X'):
            return 'X' if not self._is_winner('O') else None
        else:
            return 'O' if self._is_winner('O') else None

    def is_terminal(self):
        return '.' not in self.state or self._winner()

TicTacToeNode.initialize_partitions()


# def rotate(board):
#     """Rotate the board 90 degrees clockwise."""
#     return [list(row) for row in zip(*board[::-1])]
#
#
# def reflect(board, axis):
#     """Reflect the board along the specified axis."""
#     if axis == "horizontal":
#         return board[::-1]
#     elif axis == "vertical":
#         return [row[::-1] for row in board]
#     elif axis == "diagonal_main":
#         return [list(row) for row in zip(*board)]
#     elif axis == "diagonal_secondary":
#         return [list(row[::-1]) for row in zip(*board[::-1])]
#     else:
#         raise ValueError("Invalid reflection axis")
#
#
# def is_transformable(board1, board2):
#     """Check if board1 can be transformed into board2 by rotation or reflection."""
#     transformations = [board1]
#
#     # Generate rotated versions
#     for _ in range(3):
#         board1 = rotate(board1)
#         transformations.append(board1)
#
#     # Generate reflected versions
#     for axis in ["horizontal", "vertical", "diagonal_main", "diagonal_secondary"]:
#         transformations.append(reflect(board1, axis))
#
#     return board2 in transformations
#
#
# # Example usage
# board_a = [
#     ["X", "O", "X"],
#     ["O", "X", "O"],
#     ["X", "O", "X"]
# ]
#
# board_b = [
#     ["X", "O", "X"],
#     ["X", "X", "O"],
#     ["O", "O", "X"]
# ]
#
# print(is_transformable(board_a, board_b))  # Output: False (in this case)
node = TicTacToeNode()
game = Game(min_score = 0.0, max_score = 1.0, max_levels = 9)
game.solve(node,0, 1, partitioned=True, verbose=True)
game.show_stats()





