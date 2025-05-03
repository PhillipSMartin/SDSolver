# class TicTacToeNode:
#     def __init__(self, parent=None, player=None, play=None):
#         self.boxes = parent.boxes.copy() if parent else [' '] * 9
#         if play:
#             self.boxes[play] = player
#
#     def __str__(self):
#         s = ' ' + ' | '.join(self.boxes[0:3]) + '\n-----------\n '
#         s += ' | '.join(self.boxes[3:6]) + '\n-----------\n '
#         s += ' | '.join(self.boxes[6:9]) + '\n'
#         return s
#
#     def id(self):
#         return tuple(self.boxes)
#
#     def _winner(self):
#         for n in range(0,9,3):
#             x = set(self.boxes[n:n+3])
#             if len(x) == 1 and ' ' not in x:
#                 return list(x)[0]
#         for n in range(0, 3):
#             x = set(self.boxes[n:n+9:3])
#             if len(x) == 1 and ' ' not in x:
#                 return list(x)[0]
#         x = set(self.boxes[0:12:4])
#         if len(x) == 1 and ' ' not in x:
#             return list(x)[0]
#         x = set(self.boxes[2:8:2])
#         if len(x) == 1 and ' ' not in x:
#             return list(x)[0]
#         return None

# node1 = TicTacToeNode()
# print(node1.label())
# node2 = TicTacToeNode(node1, 'X', 4)
# print(node2.label())