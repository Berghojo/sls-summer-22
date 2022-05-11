from sls.agents import AbstractAgent
import numpy as np
import tensorflow as tf


class BasicAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(BasicAgent, self).__init__(screen_size)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            beacon = self._get_beacon(obs)
            if marine is None:
                return self._NO_OP
            marine_coords = self._get_unit_pos(marine)
            beacon_coords = self._get_unit_pos(beacon)
            vector = beacon_coords - marine_coords
            vector = np.sign(vector)

            for key, value in self._DIRECTIONS.items():
                if vector[0] == value[0] and vector[1] == value[1]:
                    d = key
                    return self._dir_to_sc2_action(d, marine_coords)
                return self._SELECT_ARMY
        else:
            return self._SELECT_ARMY

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
