import numpy as np
import tensorflow as tf
import datetime
import os
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
import random


class PolicyGradient:

    def __init__(self, actions, train):
        self.gamma = 0.99
        self.actions = list(actions.keys())
        self.epsilon = 1
        self.verbose = 1
        self.counter = 0
        self.train = train
        self.input_dim = 2
        self.model = self.create_model()
        if not self.train:
            path = 'models/abgabe02_aufgabe02_model_weights.h5'
            self.load_model_weights(path)
        self.exportfile = f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}_model_weights.h5'

    @staticmethod
    def custom_loss(G_action, policy_distribution):
        G, actions = G_action[:, 0], tf.cast(G_action[:, 1], tf.int32)
        indexes = tf.range(0, tf.size(actions))
        stacked_actions = tf.stack([indexes, actions], axis=1)
        policy_action = tf.gather_nd(indices=stacked_actions, params=policy_distribution)
        loss = (-tf.math.log(policy_action)) * G
        mean_loss = tf.math.reduce_mean(loss)
        return mean_loss


    def create_model(self):
        model = Sequential()
        model.add(Dense(units=128, activation='relu', input_dim=self.input_dim))
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=8, activation='softmax'))
        model.build((None, 2))
        model.compile(loss=PolicyGradient.custom_loss, optimizer=RMSprop(learning_rate=0.00025))
        return model

    def choose_action(self, s):

        s = np.reshape(s, [-1, 2])
        action_dist = self.model.predict(s)[0]
        action_id = np.random.choice(range(len(action_dist)), p=action_dist)
        action = self.actions[action_id]
        return action


    def learn(self, episode):
        episode_len = len(episode.states)
        if episode_len == 0:
            return
        G = []
        for t in range(episode_len):
            value = 0
            for k in range(t, episode_len):
                value += (self.gamma ** (k-t)) * episode.rewards[k]
            G.append([value, self.actions.index(episode.actions[t])])
        # update table
        self.model.fit(np.array(episode.states), np.array(G), verbose=self.verbose, batch_size=None)
        self.counter += 1
        self.verbose = 1
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
