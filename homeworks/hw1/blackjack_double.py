import gym
import numpy as np
from gym.utils import seeding
from gym import error, spaces, utils

from gym.envs.toy_text.blackjack import *


class BlackjackDouble(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        
        self.natural = False
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        
        if action == 1: 
            self.player.append(draw_card(self.np_random))
            
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.

        elif action == 0:
            done = True
            
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            
            reward = cmp(score(self.player), score(self.dealer))
            
            if (self.natural) & (is_natural(self.player)) & (reward == 1.):
                reward = 1.5

        # добавляем еще один action в котором можно удвоить премию
        elif action == 2:
            done = True

            self.player.append(draw_card(self.np_random))
            
            if is_bust(self.player):
                reward = -2.
            else:
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card(self.np_random))
                
                reward = cmp(score(self.player), score(self.dealer)) * 2
        
        return self._get_obs(), reward, done, {}


    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.player = draw_hand(self.np_random)
        self.dealer = draw_hand(self.np_random)
        return self._get_obs()
