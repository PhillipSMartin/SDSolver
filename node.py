from __future__ import annotations
from itertools import combinations
from trick import Trick, rho

import copy
import statistics

class Node:
    def __init__(self, player, parent = None):
        self.player = player  # 'N', 'E', 'S', or 'W'
        self.parent = parent

        self.unplayed_cards = None
        self.possible_holdings = {}
        self.score = 0
        self.scored = False
        self.expanded = False
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
        if not self.expanded:
            for card in self.holding:
                self.children.append(Choice(card, self))

    def __str__(self):
        return f"Decision node for {self.player}:\n" \
            f"  Holding: {self.holding}\n" \
            f"  Score: {self.score}\n" \
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


    def get_score(self):
        if not self.scored:
            self.expand_node()
            scores = [node.get_score() for node in self.children]
            if self.player in ['N', 'S']:
                self.score = max(scores)
            else:
                self.score = min(scores)
        print("*****")
        print(self)
        return self.score

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
        if not self.expanded:
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

    def is_terminal(self):
        return len(self.unplayed_cards) == 0

    def get_score(self):
        if not self.is_terminal() and not self.scored:
            self.expand_node()
            scores = [node.get_score() for node in self.children]
            self.score += statistics.mean(scores)
        print("*****")
        print(self)
        return self.score

