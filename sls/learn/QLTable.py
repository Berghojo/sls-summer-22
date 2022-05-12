import pandas as pd
import numpy as np
import os
import datetime


class QLTable:

    def __init__(self, actions, screen_size):
        self.state_space = 2
        self.index_values = int(screen_size / self.state_space)
        index = [f'{i} {e}' for i in range(-self.index_values, self.index_values + 1)
                 for e in range(-self.index_values, self.index_values + 1)]

        self.actions = actions
        self.epsilon = 1.0
        self.alpha = 0.1
        self.lamb = 0.9
        self.q_table = pd.DataFrame(0.0, index=index, columns=self.actions.keys()) #todo: moves always NW for multiple solutions

        self.exportfile = f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}_q_table.pkl'

    def choose_action(self, s):
        self.check_state_exist(s)

        if np.random.uniform(0,1) > self.epsilon:
            # choose best action
            action = self.q_table[self.q_table.index == s].idxmax(axis=1)[0]
        else:
            # choose random action
            action = np.random.choice(list(self.actions.keys()))

        return action

    def learn(self, s, a, s_new, obs):
        if s_new != 'target':
            value = self.alpha * (self.lamb * self.q_table[self.q_table.index == s].max(axis=1)[0] - self.q_table.at[s, a])
        else:
            value = self.alpha * (obs.reward - self.q_table.at[s, a])

        # update
        self.q_table.at[s, a] += value

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            print("index missing" + state)
            # append new state to q table
            #self.q_table = pd.concat([self.q_table,
             #   pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)],axis=0)

    def save_qtable(self, path):
        filename = path + self.exportfile
        self.q_table.to_pickle(filename)
        print('saved')

    def load_qtable(self, filepath):
        if os.path.isfile(filepath):
            self.q_table = pd.read_pickle(filepath)
            print('loaded')
