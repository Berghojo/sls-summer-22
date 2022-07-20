import time

from absl import app
from numpy.random import random

from sls import Env, Runner
from multiprocessing import Process
from sls.agents import *

class A2C_Worker(Process):

    def __init__(self, _CONFIG, connection, a2c):
        super(A2C_Worker, self).__init__()
        self.a2c = a2c
        self.connection = connection
        self._CONFIG = _CONFIG
        self.agent = self._CONFIG['agent'](
            train=self._CONFIG['train'],
            screen_size=self._CONFIG['screen_size'],
            connection=self.connection,
            a2c=self.a2c
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
            if recv_msg == "STEP":
                self.step()
            elif recv_msg == "RESET":
                self.reset_env()
            elif recv_msg == "CLOSE":
                self.close_env()
                break

    def reset_env(self):
        print("reset")

    def close_env(self):
        print("close")

    def step(self):
        # receive NET
        # do step
        # send Update
        self.counter += 1
        self.connection.send(("did step " + str(self.counter)))
        time.sleep(random())
