from sls.agents import AbstractAgent
from sls.learn import ExperienceReplay, CNNDuelingDDQNPrioritized
import tensorflow as tf
import numpy as np
from sls.prio_exp_replay import PrioritizedReplayBuffer


class CNN_Agent(AbstractAgent):

    def __init__(self, train, screen_size):
        tf.compat.v1.disable_eager_execution()
        super(CNN_Agent, self).__init__(screen_size)
        self.train = train
        self.last_state = None
        self.last_action = None
        self.decay_episodes = 500
        self.exp_replay = PrioritizedReplayBuffer(100_000, 0.4)
        self.min_batch_size = 6000
        self.dqn_network = CNNDuelingDDQNPrioritized(self._DIRECTIONS, train)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            beacon = self._get_beacon(obs)
            marine_coords = self._get_unit_pos(marine) # walks in 3px steps
            beacon_coords = self._get_unit_pos(beacon)
            if marine is None:
                return self._NO_OP

            state = np.array(self.get_state(obs))
            done = obs.reward == 1 or obs.last()
            if self.last_state is not None and self.train:
                self.exp_replay.add(self.last_state, self.last_action, obs.reward, state, done)
            direction = self.dqn_network.choose_action_double(state)

            if self.last_state is not None and self.train and self.exp_replay.__len__() > self.min_batch_size:
                self.dqn_network.learn(self.exp_replay)

            if done:
                self.last_action = None
                self.last_state = None
            else:
                self.last_action = direction
                self.last_state = state

            return self._dir_to_sc2_action(direction, marine_coords)
        else:
            return self._SELECT_ARMY

    def get_state(self, obs):
        return obs.observation.feature_screen.unit_density.reshape([self.screen_size, self.screen_size, 1])

    def update_target_model(self):
        print('reset networks')
        self.dqn_network.reset_q()

    def save_model(self, path):
        self.dqn_network.save_model_weights(path)

    def load_model(self, filename):
        self.dqn_network.load_model_weights(filename)

    def update_epsilon(self, episodes):
        epsilon = self.dqn_network.epsilon - (0.95 / self.decay_episodes)
        #TODO: remove hard coding
        if epsilon < 0.05:
            epsilon = 0.05
        #update epsilon
        self.dqn_network.epsilon = epsilon

    def get_epsilon(self):
        return self.dqn_network.epsilon
