from node import Node, Decision, Choice
from itertools import combinations

deck = ('a', 'k', 'q', '6', '5', '4', '3', '2')
dummy = ('a', 'q')
holding = ( '3', '2' )

node = Decision.from_setup('S', holding, deck, dummy)
print(node.get_score())
# for n in range(10):
#     node.expand_node()
#     print(f'\n***Node {n}***')
#     print(node)
#     node = node.children[0]
#
# node.expand_node()
# print('\n***Node 10***')
# print(node)




