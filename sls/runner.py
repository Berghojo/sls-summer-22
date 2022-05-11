import datetime
import os

import numpy as np
import tensorflow as tf


class Runner:
    def __init__(self, agent, env, train, load_path):
        tf.compat.v1.disable_eager_execution()
        self.agent = agent
        self.env = env
        self.train = train  # run only or train_model model?
        self.score_batch = []
        self.score = 0  # store for the scores of an episode
        self.episode = 1  # episode counter

        self.path = './graphs/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else 'run_') \
                    + type(agent).__name__

        # Tensorflow 1.X
        #self.writer = tf.summary.FileWriter(self.path, tf.get_default_graph())
        # Tensorflow 2.X mit ausgeschalteter eager_execution
        # Alle weiteren tf.summary Aufrufe müssen durch tf.compat.v1.summary tf.compat.v1.summary ersetzt werden
        self.writer = tf.compat.v1.summary.FileWriter(self.path, tf.compat.v1.get_default_graph())

        if not self.train and load_path is not None and os.path.isdir(load_path):
                self.agent.load_model(load_path)

    def summarize(self):
        # self.writer.add_summary(tf.Summary(
        #     value=[tf.Summary.Value(tag='Score per Episode', simple_value=self.score)]),
        #     self.episode
        # )
        if len(self.score_batch) < 50:
            self.score_batch.append(self.score)
        else:
            self.score_batch.pop(0)
            self.score_batch.append(self.score)
            mean = np.mean(self.score_batch)
            self.writer.add_summary(tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag='Average (50) score', simple_value=mean)]),
                self.agent.get_epsilon())
            print('Mean Score(50): ', mean)
        self.writer.add_summary(tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag='Score per Episode', simple_value=self.score)]),
                                self.episode)
        self.writer.add_summary(tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag='Epsilon', simple_value=self.score)]),
            self.agent.get_epsilon())

        # with self.writer.as_default():
        #     tf.summary.scalar('Score per Episode', self.score, step=self.episode)
        if self.train and self.episode % 10 == 0:
            self.agent.save_model(self.path)
            try:
                self.agent.update_target_model()
            except AttributeError:
                ...
        self.episode += 1
        self.score = 0

    def run(self, episodes):
        while self.episode <= episodes:
            obs = self.env.reset()
            while True:
                action = self.agent.step(obs)
                if obs.last():
                    self.agent.update_epsilon(episodes)
                    break
                obs = self.env.step(action)
                self.score += obs.reward
            self.summarize()
