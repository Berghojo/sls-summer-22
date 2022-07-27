import time

from absl import app
from numpy.random import random

from sls import Env, Runner
from multiprocessing import Process
from sls.agents import *

class A2C_Worker(Process):

    def __init__(self, worker_id, _CONFIG, connection):
        super(A2C_Worker, self).__init__()
        self.id = worker_id
        self.connection = connection
        self.obs = None
        self._CONFIG = _CONFIG
        self.agent = None
        self.env = None
        self.counter=0
        self.score = 0

    def startup(self):
        print('creating_agent', self.id)
        self.agent = self._CONFIG['agent'](
            train=self._CONFIG['train'],
            screen_size=self._CONFIG['screen_size'],
            connection=self.connection,
            worker_id=self.id
        )
        print('creating_environment', self.id)
        self.env = Env(
            screen_size=self._CONFIG['screen_size'],
            minimap_size=self._CONFIG['minimap_size'],
            visualize=self._CONFIG['visualize']
        )

    def run(self):
        self.startup()
        self.connection.send('RDY')
        while True:
            recv_msg = self.connection.recv()
            if recv_msg[0] == "STEP":
                self.step()
            elif recv_msg[0] == "RESET":
                self.connection.send(self.score)
                self.reset_env()
                self.score = 0
                self.connection.send('RDY')
            elif recv_msg[0] == "CLOSE":
                self.close_env()
                break
            else:
                print("UNKNOWN Message: ", recv_msg[0])

    def reset_env(self):
        self.obs = self.env.reset()

    def close_env(self):
        pass

    def step(self):
        action = self.agent.step(self.obs)
        if not self.obs.last():
            self.obs = self.env.step(action)
        self.score += self.obs.reward
        # if self.obs.last():
        #     print(self.id, ' finished')
