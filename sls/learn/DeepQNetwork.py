import numpy as np
import h5py
import datetime
import os
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import random

class DeepQNetwork:

    def __init__(self, actions, train):
        self.gamma = 0.9
        self.actions = list(actions.keys())
        self.epsilon = 1
        self.verbose = 1
        self.counter = 0
        self.train = train
        self.input_dim = 2
        self.model = self.create_model()
        if not self.train:
            path = 'models/abgabe03_aufgabe01_model_weights.h5'
            self.load_model_weights(path)

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.exportfile = f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}_model_weights.h5'

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=16, activation='relu', input_dim=self.input_dim))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='linear'))
        model.build((None, 2))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=0.00025))
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
            s = np.reshape(s, [-1, 2])
            action_id = np.argmax(self.model.predict(s))
            action = self.actions[action_id]
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, exp_replay):
        mini_batch = np.random.randint(exp_replay.__len__(), size=32)
        x_train = np.array(exp_replay.states)[mini_batch]
        y_train = self.model.predict(x_train)
        next_q_values = self.target_model.predict(np.array(exp_replay.states_next)[mini_batch])
        for i, idx in enumerate(mini_batch):
            if exp_replay.dones[idx]:
                value = exp_replay.rewards[idx]
            else:
                value = exp_replay.rewards[idx] + self.gamma * max(next_q_values[i])
            y_train[i][self.actions.index(exp_replay.actions[idx])] = value
        # update table
        self.model.fit(x_train, y_train, verbose=self.verbose)
        self.counter += 1
        self.verbose = 0
        if self.counter % 200 == 0:
            self.verbose = 1

    def save_model_weights(self, path):
        filename = path + self.exportfile
        self.model.save_weights(filename)
        print('saved')

    def load_model_weights(self, filepath):
        if os.path.isfile(filepath):
            print('loaded')
            self.model.load_weights(filepath)

    def reset_q(self):
        self.target_model.set_weights(self.model.get_weights())