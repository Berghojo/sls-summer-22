import numpy as np
import tensorflow as tf
import datetime
import os
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Lambda, Conv2D, Flatten, Input, Softmax, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adam
import random


class A2C_FC_PolicyGradient:

    def __init__(self, train):
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
        self.build_fit()
        if not self.train:
            path = 'models/abgabe05_aufgabe02_model_weights.h5'
            self.load_model_weights(path)
        self.exportfile = f'models/{datetime.datetime.now().strftime("%y%m%d_%H%M")}_model_weights.h5'

    def build_fit(self):
        actions = K.placeholder()
        G = K.placeholder()
        #G, actions = advantage_action[:, 0], tf.cast(advantage_action[:, 1], tf.int32)
        predictions = self.model.output
        policy, critic_value = predictions[0], predictions[1]
        policy = tf.clip_by_value(policy, 0.00001, 0.99999)
        # stacked_actions = tf.stack([indexes, actions], axis=1)
        one_hot = K.one_hot(K.cast(actions, dtype='int32'), 256)
        policy_action = tf.reduce_sum(one_hot * policy, axis=-1)
        clipped_policy_action = tf.clip_by_value(policy_action, 1e-5, 1 - 1e-5)
        advantage = G - critic_value[:, 0]
        policy_loss = -tf.math.reduce_mean(tf.stop_gradient(advantage) * (tf.math.log(policy_action)))

        value_loss = tf.math.reduce_mean(K.pow(advantage, 2))

        entropy = -1.0 * tf.math.reduce_sum(policy * tf.math.log(policy), -1)
        entropy_loss = -1.0 * tf.math.reduce_mean(entropy)
        loss = policy_loss + self.value_const * value_loss + self.entropie_const * entropy_loss
        optimizer = Adam(lr=self.learning_rate)
        update = optimizer.get_updates(loss=loss,
                                       params=self.model.trainable_weights)

        self.fit = K.function(
            inputs=[self.model.input, actions, G],
            outputs=[loss],
            updates=update
        )
        return loss

    def create_model(self):
        inputs = Input(shape=(16, 16, 1), name="img")
        l1 = Conv2D(16, (5, 5), strides=1, padding="same", activation="relu")(inputs)
        l2 = Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(l1)

        actor_conv = Conv2D(1, (1, 1), strides=1, padding="same", activation="linear")(l2)
        actor_flatten = Flatten()(actor_conv)
        actor = Activation("softmax")(actor_flatten)

        critic_flatten = Flatten()(l2)
        fc_layer = Dense(256, activation="relu")(critic_flatten)
        critic = Dense(1, activation="linear", name="critic_out")(fc_layer)

        model = Model(inputs=inputs,
                      outputs=[actor, critic],
                      name='A2C')
        #loss = self.custom_loss(y)
        ##model.compile(loss=self.custom_loss, optimizer=Adam(learning_rate=self.learning_rate))
        # model.summary()
        return model

    def choose_action(self, s):
        s = np.array(s).reshape([-1, 16, 16, 1])
        action_dists, values = self.model.predict(s)
        if np.any(action_dists <= 0):
            print('dist', action_dists)
        actions = []
        for a in action_dists:
            action_id = np.random.choice(range(len(a)), p=a)
            actions.append(action_id)
        return np.array(actions), values

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
            self.mini_batch.append([sar_batch.states[idx], np.array([value]), sar_batch.actions[idx]])
            if len(self.mini_batch) == self.mini_batch_size:
                self.learn()
                self.mini_batch = []

    def learn(self):
        states = [el[0] for el in self.mini_batch]
        actions = [el[2] for el in self.mini_batch]
        #hot_act = to_categorical(actions, 256)
        G = np.array([el[1] for el in self.mini_batch])
        self.fit([np.array(states), actions, G])

        self.counter += 1
        self.verbose = 0
        if self.counter % 200 == 0:
            self.verbose = 1

    def save_model_weights(self):
        filename = self.exportfile
        self.model.save_weights(filename)
        print('saved')

    def load_model_weights(self, filepath):
        if os.path.isfile(filepath):
            print('loaded')
            self.model.load_weights(filepath)
