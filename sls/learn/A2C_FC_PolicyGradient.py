import numpy as np
import tensorflow as tf
import datetime
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, Conv2D, Flatten, Input, Softmax
from tensorflow.keras.optimizers import RMSprop
import random


class A2C_FC_PolicyGradient:

    def __init__(self, actions, train):
        # Definitions
        self.n_step_return = 5
        self.value_const = 0.5
        self.entropie_const = 0.005
        self.learning_rate = 0.0007
        # TODO: clean up
        self.mini_batch = []
        self.mini_batch_size = 64
        self.gamma = 0.99
        self.verbose = 1
        self.counter = 0
        self.train = train
        self.input_dim = 2
        self.model = self.create_model()
        if not self.train:
            path = 'models/abgabe05_aufgabe01_model_weights.h5'
            self.load_model_weights(path)
        self.exportfile = f'{datetime.datetime.now().strftime("%y%m%d_%H%M")}_model_weights.h5'

    def custom_loss(self, advantage_action, model_output):
        G, actions = advantage_action[:, 0], tf.cast(advantage_action[:, 1], tf.int32)
        policy, critic_value = model_output[:, :-1], model_output[:, -1]
        indexes = tf.range(0, tf.size(actions))
        stacked_actions = tf.stack([indexes, actions], axis=1)
        policy_action = tf.gather_nd(indices=stacked_actions, params=policy)
        advantage = G - critic_value
        policy_loss = -tf.math.reduce_mean(tf.stop_gradient(advantage) * (tf.math.log(policy_action)))
        value_loss = tf.math.reduce_mean(advantage ** 2)
        entropy = tf.math.negative(tf.math.reduce_sum(policy * tf.math.log(policy), 1))
        entropy_loss = -tf.math.reduce_mean(entropy)
        loss = policy_loss + self.value_const * value_loss + self.entropie_const * entropy_loss
        return loss

    def create_model(self):
        inputs = Input(shape=(16, 16, 1), name="img")
        l1 = Conv2D(16, (5, 5), strides=1, padding="same", activation="relu")(inputs)
        l2 = Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(l1)

        actor_conv = Conv2D(1, (1, 1), strides=1, padding="same", activation="linear")(l2)
        actor_flatten = Flatten()(actor_conv)
        #test_fc_layer = Dense(256, activation="relu")(actor_flatten)
        actor = Softmax(name='softmax_actor')(actor_flatten)
        #actor = Dense(256, activation='softmax')(actor_flatten)
        #actor = Dense(8, activation="softmax", name="actor_test")(test_fc_layer)

        critic_flatten = Flatten()(l2)
        fc_layer = Dense(256, activation="relu")(critic_flatten)
        #actor_test = Dense(256, activation="softmax", name="actor_out")(fc_layer)
        critic = Dense(1, activation="linear", name="critic_out")(fc_layer)
        prediction = tf.concat([actor, critic], 1)
        model = Model(inputs=inputs,
                      outputs=prediction,
                      name='A2C')
        model.compile(loss=self.custom_loss, optimizer=RMSprop(learning_rate=self.learning_rate))
        # model.summary()
        return model

    def choose_action(self, s):
        s = s.reshape([-1, 16, 16, 1])
        prediction = self.model.predict(s)

        action_dist, value = prediction[0, :-1], prediction[0, -1]
        direction_key = np.random.choice(range(len(action_dist)), p=action_dist)
        return direction_key, value

    def add_to_batch(self, sar_batch):
        value = 0
        for k in range(0, self.n_step_return):
            value += (self.gamma ** k) * sar_batch.rewards[k]
        value += (self.gamma ** self.n_step_return) * sar_batch.values[self.n_step_return]
        self.mini_batch.append([sar_batch.states[0], value, sar_batch.actions[0]])
        if len(self.mini_batch) == self.mini_batch_size:
            self.learn()
            self.mini_batch = []

    def add_last_to_batch(self, sar_batch):
        for idx, el in enumerate(sar_batch.states):
            value = 0
            for k in range(0, len(sar_batch.states) - idx):
                if k == self.n_step_return:
                    value += (self.gamma ** self.n_step_return) * sar_batch.values[self.n_step_return]
                else:
                    value += (self.gamma ** k) * sar_batch.rewards[k + idx]
            self.mini_batch.append([sar_batch.states[idx], value, sar_batch.actions[idx]])
            if len(self.mini_batch) == self.mini_batch_size:
                self.learn()
                self.mini_batch = []

    def learn(self):
        states = [el[0] for el in self.mini_batch]
        G = [[el[1], el[2]] for el in self.mini_batch]
        self.model.fit(np.array(states), np.array(G), verbose=self.verbose, batch_size=None)
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
