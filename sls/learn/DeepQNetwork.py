import numpy as np
import h5py
import datetime
import os
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

class DeepQNetwork:

    def __init__(self, actions, train):
        self.gamma = 0.9
        self.actions = actions
        self.epsilon = 1
        self.model = Sequential()
        self.model.add(Dense(units=16, activation='relu', input_dim=2))
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dense(units=8, activation='linear'))
        self.model.compile(loss='mse', optimizer=RMSprop(lr=0.00025))
        self.target_model = clone_model(self.model)
        self.target_model.build((None, 2))  # number of variables in input layer
        self.target_model.compile(loss='mse', optimizer=RMSprop(lr=0.00025))
        self.target_model.set_weights(self.model.get_weights())
        self.exportfile = f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}_model_weights.h5'
        self.directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    def choose_action(self, s):
        if np.random.uniform(0, 1) > self.epsilon:
            # choose best action (random selection if multiple solutions)
            s = np.reshape(s, [-1, 2])
            action_id = np.argmax(self.model.predict(s))
            action = list(self.actions.keys())[action_id]
        else:
            # choose random action
            action = np.random.choice(list(self.actions.keys()))

        return action

    def learn(self, exp_replay):
        mini_batch = np.random.randint(len(exp_replay), size=32).tolist()
        x_train = np.array(exp_replay.states)[mini_batch]
        y_train = self.target_model.predict(x_train)
        next_q_values = self.target_model.predict(np.array(exp_replay.states_next)[mini_batch])
        i = 0
        for idx in mini_batch:

            if exp_replay.dones[idx]:
                value = exp_replay.rewards[idx]
            else:
                next_state = exp_replay.states_next[idx].reshape([-1, 2])
                value = exp_replay.rewards[idx] + self.gamma * np.max(next_q_values[i])
            y_train[i, self.directions.index(exp_replay.actions[idx])] = value
            i += 1
        # update table
        self.model.fit(x_train, y_train, verbose=False)

    def save_model_weights(self, path):
        filename = path + self.exportfile
        self.model.save_weights(filename)
        print('saved')

    def load_model_weights(self, filepath):
        if os.path.isfile(filepath):
            self.model = h5py.File(filepath, 'r')
            print('loaded')