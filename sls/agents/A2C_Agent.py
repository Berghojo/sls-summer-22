from sls.agents import AbstractAgent
from sls.learn import ExperienceReplay, A2C_PolicyGradient, State_Batch
import tensorflow as tf
import numpy as np


class A2C_Agent(AbstractAgent):

    def __init__(self, train, screen_size):
        tf.compat.v1.disable_eager_execution()
        super(A2C_Agent, self).__init__(screen_size)
        self.train = train
        self.last_state = None
        self.last_action = None
        self.decay_episodes = 500
        self.sar_batch = State_Batch()
        self.value = 0
        self.neg_reward = -0.01
        self.pos_reward = 1
        self.n_step_return = 5
        self.a2c = A2C_PolicyGradient(self._DIRECTIONS, train)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            beacon = self._get_beacon(obs)
            marine_coords = self._get_unit_pos(marine)
            beacon_coords = self._get_unit_pos(beacon)
            if marine is None:
                return self._NO_OP
            state = self.get_state(obs)
            reward = self.pos_reward if obs.reward == 1 else self.neg_reward
            done = obs.reward == 1 or obs.last()

            if self.last_state is not None and self.train:
                self.sar_batch.add_step(self.last_state, self.last_action, reward, self.value)

            direction, self.value = self.a2c.choose_action(state)
            self.last_action = direction
            self.last_state = state

            if done and self.train:
                self.a2c.add_last_to_batch(self.sar_batch)
                self.sar_batch.clear()
                self.last_action = None
                self.last_state = None
            if self.train and (len(self.sar_batch.states) >= self.n_step_return+1):
                self.a2c.add_to_batch(self.sar_batch)
                self.sar_batch.pop(0)
            return self._dir_to_sc2_action(direction, marine_coords)
        else:
            return self._SELECT_ARMY

    def get_state(self, obs):
        return np.array(obs.observation.feature_screen.unit_density.reshape([self.screen_size, self.screen_size, 1]))

    def update_target_model(self):
        print('reset networks')
        self.a2c.reset_q()

    def save_model(self, path):
        self.a2c.save_model_weights(path)

    def load_model(self, filename):
        self.a2c.load_model_weights(filename)

    def update_epsilon(self, episodes):
        epsilon = self.a2c.epsilon - (0.95 / self.decay_episodes)
        if epsilon < 0.05:
            epsilon = 0.05
        # update epsilon
        self.a2c.epsilon = epsilon

    def get_epsilon(self):
        return self.a2c.epsilon
