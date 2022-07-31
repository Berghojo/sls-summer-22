from sls.agents import AbstractAgent
from sls.learn import ExperienceReplay, A2C_FC_PolicyGradient, State_Batch
import tensorflow as tf
import numpy as np


class A2C_FC_Agent(AbstractAgent):

    def __init__(self, train, screen_size, connection, worker_id):
        tf.compat.v1.disable_eager_execution()
        super(A2C_FC_Agent, self).__init__(screen_size)
        self.train = train
        self.last_state = None
        self.last_action = None
        self.decay_episodes = 500
        self.sar_batch = State_Batch()
        self.value = 0
        self.neg_reward = -0.01
        self.connection = connection
        self.id = worker_id
        self.pos_reward = 1
        self.n_step_return = 5
        self.switch = False
        self.counter = 0

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            if marine is None:
                self.connection.send([None, None])
                self.connection.send([None, None, None])
                return self._NO_OP
            state = self.get_state(obs)
            reward = self.pos_reward if obs.reward == 1 else self.neg_reward
            done = obs.reward == 1 or obs.last()

            if self.last_state is not None and self.train:
                self.sar_batch.add_step(self.last_state, self.last_action, reward, self.value)
            #directions = list(self._DIRECTIONS.keys())
            #direction_key, self.value = self.a2c.choose_action(state)
            self.connection.send([self.id, state])
            action_key, self.value = self.connection.recv()
            pixel = np.divmod(action_key, self.screen_size)
            self.last_action = action_key
            self.last_state = state
            if done and self.train:
                self.connection.send([self.sar_batch, done, obs.last()])
                self.sar_batch.clear()
                self.last_action = None
                self.last_state = None
            elif self.train and (len(self.sar_batch.states) >= self.n_step_return + 1):
                self.connection.send([self.sar_batch, done, obs.last()])
                self.sar_batch.pop(0)
            else:
                self.connection.send([None, None, None])

            return self._MOVE_SCREEN("now", pixel)
            #return self._dir_to_sc2_action(directions[direction_key], marine_coords)
        else:
            self.connection.send([None, None])
            self.connection.send([None, None, None])
            return self._SELECT_ARMY

    def get_state(self, obs):
        return np.array(obs.observation.feature_screen.unit_density.reshape([self.screen_size, self.screen_size, 1]))
