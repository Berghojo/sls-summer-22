import pandas as pd
import os
import random

class QTableLearn:
    def __int__(self):
        self.state_space = 8
        index = [f'{i} {e}' for i in range(-self.state_space, self.state_space + 1)
                 for e in range(-self.state_space, self.state_space + 1)]

        self.epsilon = 0.99
        self.alpha = 0.5
        self.lamb = 0.9
        self.q_table = pd.DataFrame(0.0, index=index, columns=self._DIRECTIONS.keys())

    def choose_action(self, s):
        self.check_state_exist(s)

        if random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[s, :]

            # some actions have the same value
            state_action = state_action.reindex(random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = random.choice(list(self._DIRECTIONS.keys()))

        return action

    def learn(self, s, a, r, s_new, obs):
        if s_new != 'target':
            value = self.alpha * (self.lamb * self.q_table.ix[s_new, :].max() -
                                  self.q_table.at[self.last_state, self.last_action])


        else:
            value = self.alpha * (obs.reward - self.q_table.at[self.last_state, self.last_action])
            self.last_action = None
            self.last_state = None
        self.q_table.at[self.last_state, self.last_action] += value

    def get_state(self, beacon, marine, obs):
        marine_coords = self._get_unit_pos(marine)
        beacon_coords = self._get_unit_pos(beacon)
        distance = beacon_coords - marine_coords
        state = (distance / self.state_space).astype(int)
        state_string = f'{state[0]} {state[1]}'
        if obs.reward != 0:
            state_string = 'target'

        if state_string == self.last_state:
            print('no transition')
        return marine_coords, state_string

    def save_qtable(self):

    def load_qtable(self):
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')