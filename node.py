from __future__ import annotations
from itertools import combinations
from trick import Trick, rho

import copy

class Node:
    def __init__(self, player, parent = None):
        self.player = player  # 'N', 'E', 'S', or 'W'
        self.parent = parent

        self.unplayed_cards = None
        self.possible_holdings = {}
        self.children = []

        if parent and parent.trick.winner is None:
            # copy an incomplete trick
            self.trick = copy.deepcopy(parent.trick)
        else:
            # else start a new one
            self.trick = Trick(player)

class Decision(Node):
    def __init__(self, player, holding:tuple, parent=None):
        super().__init__(player, parent=parent)
        self.holding = holding
        self.unplayed_cards = copy.copy(parent.unplayed_cards) if parent else None
        self.possible_holdings = copy.copy(parent.possible_holdings) if parent else None

    @classmethod
    def from_setup(cls, player, holding:tuple, deck:tuple, dummy:tuple):
        node = cls(player, holding)

        node.dummy = dummy
        node.unplayed_cards = set(deck) - set(dummy)
        node.possible_holdings = {
            'N': [ dummy ],
            'E': list(combinations(node.unplayed_cards, len(dummy))),
            'S': list(combinations(node.unplayed_cards, len(dummy))),
            'W': list(combinations(node.unplayed_cards, len(dummy))),
        }
        node.trick = Trick(player)

        return node

    @classmethod
    def from_play(cls, player, holding:tuple, parent:Choice):
        node = cls(player, holding, parent=parent)
        return node

    def expand_node(self):
        for card in self.holding:
            self.children.append(Choice(card, self))

    def __str__(self):
        return f"Decision node for {self.player}:\n" \
            f"  Holding: {self.holding}\n" \
            f"  Unplayed cards: {self.unplayed_cards}\n" \
            f"  North: ({len(self.possible_holdings['N'])}): {self.possible_holdings['N']}\n" \
            f"  East ({len(self.possible_holdings['E'])}): {self.possible_holdings['E']}\n" \
            f"  South ({len(self.possible_holdings['S'])}): {self.possible_holdings['S']}\n" \
            f"  West ({len(self.possible_holdings['W'])}): {self.possible_holdings['W']}\n" \
            f"  Trick: {self.trick.cards_played}\n" \
            f"  Number of choices: {len(self.children)}"

    def show_choices(self):
        for index, choice in enumerate(self.children):
            print(f"Choice {index}: {choice}")

class Choice(Node):
    def __init__(self, play, parent:Decision):
        super().__init__(parent.player, parent=parent)
        self.trick.play_card(play)
        self.unplayed_cards = {card for card in parent.unplayed_cards if card != play}
        self.possible_holdings[parent.player] = [tuple(card for card in item if card != play)
            for item in self.parent.possible_holdings[parent.player] if play in item]
        if parent.player != 'N':
            self.possible_holdings['N'] = self.parent.possible_holdings['N'].copy()
        if parent.player != 'E':
            self.possible_holdings['E'] = [item for item in self.parent.possible_holdings['E'] if play not in item]
        if parent.player != 'S':
            self.possible_holdings['S'] = [item for item in self.parent.possible_holdings['S'] if play not in item]
        if parent.player != 'W':
            self.possible_holdings['W'] = [item for item in self.parent.possible_holdings['W'] if play not in item]
        self.score = 1  if self.trick.winner in ['N', 'S'] else 0

    def expand_node(self):
        if self.trick.winner:
            next_to_play = self.trick.winner
        else:
            next_to_play = rho(self.player)

        for holding in self.possible_holdings[next_to_play]:
            if set(holding).isdisjoint(self.parent.holding):
                self.children.append(Decision(next_to_play, holding, self))

    def __str__(self):
        return f"Choice node for {self.player}, play: {self.trick.cards_played[-1]}\n" \
            f"  Score: {self.score}\n" \
            f"  Trick: {self.trick.cards_played}\n" \
            f"  Number of scenarios: {len(self.children)}"

    def show_scenarios(self):
        for index, scenario in enumerate(self.children):
            print(f"Scenario {index}: {scenario}")

    #         @classmethod
    # def from_play(cls, player, play, parent):
    #     node = cls(1, player, parent=parent)
    #     node.trick = copy.copy(parent.trick)
    #     winner = node.trick.play_card(play)
    #     if winner:
    #         node.quit_trick()
    #     return node
    #
    #
    #
    #
    #
    # def quit_trick(self):
    #     pass
    #
    # def expand_node(self):
    #     pass

    # def is_terminal(self):
    #     return len(self.holding) == 1 and len(self.current_trick) == 2
    #
    # def expand_node(self):
    #     rho_possible_holdings = [rho_holding for rho_holding in self.possible_holdings[rho(self.player)]
    #                              if rho_holding.isdisjoint(self.holding)]
    #     for card in self.holding:
    #         for rho_holding in rho_possible_holdings:
    #             self.children.append(Node(rho(self.player), rho_holding, self ))
    #
    # def score_trick(self):
    #     # determine winner
    #     winner = min(self.current_trick, key=lambda item: Node.deck.index(item))
    #
    #     # increment ns score if ns won the trick
    #     if not ((self.current_trick.index(winner) % 2) ^ (Node.seats.index(self.trick_leader) % 2)):
    #         self.ns_tricks_won_so_far += 1
    #
    #     # return current score
    #     return self.ns_tricks_won_so_far
    #
    # def max_value(self):
    #     if self.is_terminal():
    #         self.play(self.holding[0])
    #         self.play(self.possible_holdings[rho(self.player)][0][0])
    #         return self.score_trick()
    #     else:
    #         self.expand_node()
    #         v = 0
    #         for child in self.children:
    #             v = max(v, child.score)
    #         return v
    #
    # def min_value(self):
    #     if self.is_terminal():
    #         self.play(self.holding[0])
    #         self.play(self.possible_holdings[rho(self.player)][0][0])
    #         return self.score_trick()
    #     else:
    #         self.expand_node()
    #         v = math.inf
    #         for child in self.children:
    #             v = min(v, child.score)
    #         return v

