from absl import app
from sls import Env, Runner
from sls.agents import *
from multiprocessing import Process, Pipe

_CONFIG = dict(
    episodes=10000,
    screen_size=16,
    minimap_size=64,
    visualize=True,
    train=True,
    agent=A2C_FC_Agent,
    load_path='./graphs/...',
    worker=2
)


def main(unused_argv):

    agent = [_CONFIG['agent'](
        train=_CONFIG['train'],
        screen_size=_CONFIG['screen_size']
    ) for _ in range(_CONFIG['worker'])]

    env = [Env(
        screen_size=_CONFIG['screen_size'],
        minimap_size=_CONFIG['minimap_size'],
        visualize=_CONFIG['visualize']
    )for _ in range(_CONFIG['worker'])]

    runner = Runner(
        agent= agent,
        env=env,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)
