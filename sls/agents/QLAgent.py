from sls.agents import AbstractAgent
import numpy as np
import pandas as pd
import tensorflow as tf


class QLAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(QLAgent, self).__init__(screen_size)
        self.state_space = 8
        index = [f'{i} {e}' for i in range(-self.state_space, self.state_space)
                 for e in range(-self.state_space, self.state_space)]

        self.q_table = pd.DataFrame(0, index=index, columns= self._DIRECTIONS.keys())
        print(self.q_table)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            beacon = self._get_beacon(obs)
            if marine is None:
                return self._NO_OP
            marine_coords = self._get_unit_pos(marine)
            beacon_coords = self._get_unit_pos(beacon)
            distance = beacon_coords - marine_coords
            state = (distance / self.state_space).astype(int)
            direction = self.q_table[f'{state[0]} {state[1]}']
            print(direction)

            return self._SELECT_ARMY
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
