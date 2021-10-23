import gym
import numpy as np
from gym.utils import seeding
from gym import error, spaces, utils

from gym.envs.toy_text.blackjack import *

# увеличиваем колоду
DECK_COUNT = 100


class BlackjackCount(gym.Env):
    def __init__(self):      
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2),
            spaces.Box(-22.0, +22.0, shape=(1,1), dtype=np.float32)))
        
        
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * DECK_COUNT
        self.card_points = {1: -1, 2: 0.5, 3: 1,4: 1, 5: 1.5, 6: 1, 7: 0.5, 8: 0, 9: -0.5, 10: -1,}
        self.card_counter = 0.0
        
        self.natural = False
        self.seed()
        self.reset()
    
    def cmp(self, a, b):
        return float(a > b) - float(a < b)


    def draw_card(self, np_random):
        card = self.deck.pop(np_random.randint(0, len(self.deck)))
        self.card_counter += self.card_points[card]
        return int(card)


    def draw_hand(self, np_random):
        return [self.draw_card(np_random), self.draw_card(np_random)]


    def usable_ace(self, hand):
        return 1 in hand and sum(hand) + 10 <= 21


    def sum_hand(self, hand):
        if self.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)


    def is_bust(self, hand):
        return self.sum_hand(hand) > 21


    def score(self, hand):
        return 0 if self.is_bust(hand) else self.sum_hand(hand)


    def is_natural(self, hand):
        return sorted(hand) == [1, 10]


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        assert self.action_space.contains(action)
        
        if action == 1:
            self.player.append(self.draw_card(self.np_random))
            
            if self.is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        
        elif action == 0:
            done = True
            
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card(self.np_random))
            
            reward = self.cmp(self.score(self.player), self.score(self.dealer))
            
            if self.natural and self.is_natural(self.player) and reward == 1.:
                reward = 1.5

        elif action == 2:
            done = True
            self.player.append(self.draw_card(self.np_random))
            
            if self.is_bust(self.player):
                reward = -2.
            else:
                while self.sum_hand(self.dealer) < 17:
                    self.dealer.append(self.draw_card(self.np_random))
                
                reward = self.cmp(self.score(self.player), self.score(self.dealer)) * 2
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (self.sum_hand(self.player), self.dealer[0], self.usable_ace(self.player), self.card_counter)

    def reset(self):
        if len(self.deck) < 15:
            self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * DECK_COUNT
            self.card_counter = 0.0
        
        self.dealer = self.draw_hand(self.np_random)
        self.player = self.draw_hand(self.np_random)
        return self._get_obs()
