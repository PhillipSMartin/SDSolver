rank = ['a', 'k', 'q', 'j', 't', '9', '8', '7', '6', '5', '4', '3', '2']

def rho(player):
    if player == 'N':
        return 'E'
    elif player == 'E':
        return 'S'
    elif player == 'S':
        return 'W'
    elif player == 'W':
        return 'N'
    else:
        raise ValueError(f'Invalid player: {player}')

class Trick:
    def __init__(self, player):
        self.next_to_play = player
        self.cards_played = []
        self.high_card = 100
        self.current_winner = None
        self.winner = None

    # adds given card to trick
    # if trick is no over returns None
    # if trick is over, return seat of winner
    def play_card(self, card):
        self.cards_played.append(card)
        if rank.index(card) < self.high_card:
            self.high_card = rank.index(card)
            self.current_winner = self.next_to_play
        if len(self.cards_played) < 4:
            self.next_to_play = rho(self.next_to_play)
            return None
        else:
            self.next_to_play = None
            self.winner = self.current_winner
            return self.winner

if __name__ == "__main__":
    trick = Trick('S')
    print(trick.play_card('2'))
    print(trick.play_card('a'))
    print(trick.play_card('k'))
    print(trick.play_card('4'))