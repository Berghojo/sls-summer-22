from absl import app
from sls import Env, Runner
from sls.agents import *

from absl import app
from sls import Env, Runner
from sls.agents import *



def worker_fkt(child_conn, a2c, _CONFIG, ):
    agent = _CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size'],
        connection=child_conn,
        a2c=a2c
    )

    env = Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )

    runner = Runner(
        agent=agent,
        env=env,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path']
    )

    runner.run(episodes=_CONFIG['episodes'])


class A2C_Worker:

    def __init__(self, _CONFIG, connection, a2c):
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
        self.run()

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
        print("step")
