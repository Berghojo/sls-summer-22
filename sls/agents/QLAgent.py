import random
import datetime
from sls.agents import AbstractAgent

import numpy as np
import pandas as pd
import tensorflow as tf


class QLAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(QLAgent, self).__init__(screen_size)
        self.state_space = 2
        self.index_values = int(screen_size / self.state_space)
        index = [f'{i} {e}' for i in range(-self.index_values, self.index_values+1)
                 for e in range(-self.index_values, self.index_values+1)]
        self.epsilon = 1
        self.train = train
        self.last_state = None
        self.last_action = None
        self.alpha = 0.1
        self.file_stamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
        self.lamb = 0.9
        if train:
            self.q_table = pd.DataFrame(0, index=index, columns= self._DIRECTIONS.keys())
            self.epsilon = 1
        else:
            self.q_table = self.load_model('220512_0052_q_table.pkl')
            self.epsilon = 0
        print(self.q_table)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            beacon = self._get_beacon(obs)
            if marine is None:
                return self._NO_OP
            marine_coords, state_string = self.get_state(beacon, marine, obs)
            directions = self.q_table[self.q_table.index == state_string]
            directions = directions.iloc[:, np.random.permutation(8)]
            direction = directions.idxmax(axis=1)[0]
            if self.last_state and self.train:
                self.update_q_table(directions, obs)
            if random.uniform(0, 1) <= self.epsilon:
                direction = random.choice(list(self._DIRECTIONS.keys()))

            self.last_action = direction
            self.last_state = state_string

            return self._dir_to_sc2_action(direction, marine_coords)
        else:
            return self._SELECT_ARMY

    def get_state(self, beacon, marine, obs):
        marine_coords = self._get_unit_pos(marine)
        beacon_coords = self._get_unit_pos(beacon)
        distance = beacon_coords - marine_coords
        state = (distance / self.state_space).astype(int)
        state = [el + 1 if el >= 0 else el for el in state]
        if obs.reward != 0:
            state = [0, 0]
        state_string = f'{state[0]} {state[1]}'
        # if state_string == self.last_state:
        #     print('no transition')
        return marine_coords, state_string

    def update_q_table(self, directions, obs):
        if obs.reward != 1 and not obs.last():
            value = self.alpha * (self.lamb * directions.max(axis=1)[0] -
                                  self.q_table.at[self.last_state, self.last_action])
            self.q_table.at[self.last_state, self.last_action] += value
        else:
            self.q_table.at[self.last_state, self.last_action] += self.alpha * \
                                                               (obs.reward -
                                                                self.q_table.at[self.last_state, self.last_action])
            self.last_action = None
            self.last_state = None

    def update_epsilon(self, episodes):
        self.epsilon -= 1/(episodes-50)
        if self.epsilon < 0.05:
            self.epsilon = 0

    def get_epsilon(self):
        return self.epsilon

    def save_model(self, path):
        filename = path + f'{self.file_stamp}_q_table.pkl'
        self.q_table.to_pickle(filename)
        print('saved')

    def load_model(self, filename):
        return pd.read_pickle('models/'+ filename)
