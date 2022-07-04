import numpy as np
import h5py
import datetime
import os
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Lambda, Conv2D, Flatten
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import random


class CNNDuelingDDQNPrioritize:

    def __init__(self, actions, train):
        self.gamma = 0.9
        self.actions = list(actions.keys())
        self.epsilon = 1
        self.verbose = 1
        self.beta = 0.6
        self.beta_inc = 5e-6
        self.epsilon_replay = 1e-6
        self.counter = 0
        self.train = train
        self.input_dim = 2
        self.model = self.create_model()
        if not self.train:
            path = 'models/cnn_model_weights.h5'
            self.load_model_weights(path)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.exportfile = f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}_model_weights.h5'

    def lambda_layer(self, input_lmd):
        value = input_lmd[:, 0]
        value = tf.reshape(value, [-1, 1])
        advantages = input_lmd[:, 1:]
        mean_advantage = tf.math.reduce_mean(advantages, axis=1)
        mean_advantage = tf.reshape(mean_advantage, [-1, 1])
        q_value = value + (advantages - mean_advantage)
        return q_value

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(16, 5, strides=1, activation='relu', input_shape=(16, 16, 1), padding='same'))
        model.add(Conv2D(32, 3, strides=1, activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=9, activation='linear'))
        model.add(Lambda(lambda x: self.lambda_layer(x)))
        model.build((None, 2))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00025))
        model.summary()
        return model

    def choose_action(self, s):
        if np.random.uniform(0, 1) > self.epsilon or not self.train:
            # choose best action (random selection if multiple solutions)
            s = np.reshape(s, [-1, 2])


            action_id = np.argmax(self.model.predict(s))
            action = self.actions[action_id]
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def choose_action_double(self, s):
        if random.randint(0, 1) == 1 and self.train:
            temp = self.model
            self.model = self.target_model
            self.target_model = temp
        if np.random.uniform(0, 1) > self.epsilon or not self.train:
            # choose best action (random selection if multiple solutions)
            s = s.reshape([-1, 16, 16, 1])
            action_id = np.argmax(self.model.predict(s))
            action = self.actions[action_id]
        else:
            # choose random action
            action = np.random.choice(self.actions)
        if self.beta < 1:
            self.beta = self.beta + float(self.beta_inc)
        else:
            self.beta = 1
        return action

    def learn(self, exp_replay):
        states, actions, rewards, next_states, dones, weights, indexes = \
            exp_replay.sample(32, self.beta)
        x_train = np.array(states)
        y_train = self.model.predict(x_train)

        next_q_values = self.target_model.predict(np.array(next_states))
        for i in range(len(actions)):
            if dones[i]:
                value = rewards[i]
            else:
                value = rewards[i] + self.gamma * max(next_q_values[i])
            y_train[i][self.actions.index(actions[i])] = value
        # update table
        self.model.fit(x_train, y_train, verbose=self.verbose, sample_weight=weights)

        self.counter += 1
        self.verbose = 0
        if self.counter % 200 == 0:
            self.verbose = 1
        priority_list = []
        Q = self.model.predict(x_train)
        target = y_train
        for i, a in enumerate(actions):
            action = self.actions.index(a)
            error = np.abs(Q[i][action] - target[i][action])
            priority = error + float(self.epsilon_replay)
            priority_list.append(priority)
        exp_replay.update_priorities(indexes, priority_list)

    def save_model_weights(self, path):
        filename = f'{path}{self.exportfile}'
        self.model.save_weights(filename)
        print('saved')

    def load_model_weights(self, filepath):
        if os.path.isfile(filepath):
            print('loaded')
            self.model.load_weights(filepath)
        #self.target_model.set_weights(self.model.get_weights())

    def reset_q(self):
        pass
