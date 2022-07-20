import time

from absl import app
from numpy.random import random

from sls import Env, Runner
from multiprocessing import Process
from sls.agents import *

class A2C_Worker(Process):

    def __init__(self, worker_id, _CONFIG, connection, a2c_net):
        super(A2C_Worker, self).__init__()
        self.id = worker_id
        self.connection = connection
        self.obs = None
        self._CONFIG = _CONFIG
        self.agent = self._CONFIG['agent'](
            train=self._CONFIG['train'],
            screen_size=self._CONFIG['screen_size'],
            connection=self.connection,
            a2c_net=a2c_net
        )

        self.env = Env(
            screen_size=self._CONFIG['screen_size'],
            minimap_size=self._CONFIG['minimap_size'],
            visualize=self._CONFIG['visualize']
        )

        self.runner = Runner(
            agent=self.agent,
            env=self.env,
            train=self._CONFIG['train'],
            load_path=self._CONFIG['load_path']
        )
        self.counter=0

    def run(self):
        while True:
            recv_msg = self.connection.recv()
            if recv_msg[0] == "STEP":
                self.step(self.obs) #TODO change to sended observer
            elif recv_msg[0] == "RESET":
                self.reset_env()
            elif recv_msg[0] == "CLOSE":
                self.close_env()
                break

    def reset_env(self):
        self.obs = self.env.reset()

    def close_env(self):
        pass

    def step(self, a2c):
        print("updated net on" + str(self.id))
        self.agent.a2c_net = a2c
        action = self.agent.step(self.obs)
        self.obs = self.env.step(action)
