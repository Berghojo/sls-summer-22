from sls.agents import AbstractAgent
from sls.learn import ExperienceReplay, State_Batch, A2C_PolicyGradient
from tensorflow.keras.layers import Dense, Lambda, Conv2D, Flatten, Input
import tensorflow as tf
import numpy as np


class A2C_Agent(AbstractAgent):

    def __init__(self, train, screen_size, connection):
        tf.compat.v1.disable_eager_execution()
        super(A2C_Agent, self).__init__(screen_size)
        self.train = train
        self.last_state = None
        self.last_action = None
        self.decay_episodes = 500
        self.sar_batch = State_Batch()
        self.connection = connection
        self.a2c = A2C_PolicyGradient(train)
        self.actions = list(self._DIRECTIONS.keys())
        self.value = 0
        self.neg_reward = -0.01
        self.pos_reward = 1
        self.n_step_return = 5

    def step(self, obs):
        print("Agent is doing a step and communication on con: ", self.connection)
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            print('getting_coords')
            marine = self._get_marine(obs)
            beacon = self._get_beacon(obs)
            marine_coords = self._get_unit_pos(marine)
            beacon_coords = self._get_unit_pos(beacon)
            if marine is None:
                print("marine is empty")
                self.connection.send([None, None, None])
                return self._NO_OP
            print('getting_state')
            state = self.get_state(obs)
            reward = self.pos_reward if obs.reward == 1 else self.neg_reward
            done = obs.reward == 1 or obs.last()

            if self.last_state is not None and self.train:
                self.sar_batch.add_step(self.last_state, self.last_action, reward, self.value)
            print('choosing')
            direction, self.value = self.choose_action(state)
            self.last_action = list(self._DIRECTIONS.keys()).index(direction)
            self.last_state = state
            print('sending')
            if done and self.train:
                self.connection.send([self.sar_batch, done, obs.last()])
                self.sar_batch.clear()
                self.last_action = None
                self.last_state = None
            elif self.train and (len(self.sar_batch.states) >= self.n_step_return+1):
                #self.a2c.add_to_batch(self.sar_batch)
                self.connection.send([self.sar_batch, done, obs.last()])
                self.sar_batch.pop(0)
            else:
                self.connection.send([None, None, None])
            return self._dir_to_sc2_action(direction, marine_coords)
        else:
            print('selecting army')
            self.connection.send([None, None, None])
            return self._SELECT_ARMY

    def choose_action(self, s):
        # s = s.reshape([-1, 16, 16, 1])
        # prediction = self.a2c_net.predict(s)
        # action_dist, value = prediction[0, :-1], prediction[0, -1]
        # if np.any(action_dist <= 0):
        #     print('dist', action_dist)
        # action_id = np.random.choice(range(len(action_dist)), p=action_dist)
        # action = self.actions[action_id]
        # return action, value
        return 'S', 1

    def get_state(self, obs):
        return np.array(obs.observation.feature_screen.unit_density.reshape([self.screen_size, self.screen_size, 1]))
