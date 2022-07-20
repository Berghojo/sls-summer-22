from absl import app
from sls import Env, Runner
from sls.agents import *

from sls.learn import A2C_PolicyGradient
from multiprocessing import Process, Pipe
from sls.worker import worker_fkt

_CONFIG = dict(
    episodes=10,
    screen_size=16,
    minimap_size=64,
    visualize=True,
    train=True,
    agent=A2C_Agent,
    load_path='./graphs/...',
)

_Worker = 2
_DIRECTIONS = {'N': [0, -1],
               'NE': [1, -1],
               'E': [1, 0],
               'SE': [1, 1],
               'S': [0, 1],
               'SW': [-1, 1],
               'W': [-1, 0],
               'NW': [-1, -1]}


def main(unused_argv):
    workers = []
    workers_process = []
    a2c = A2C_PolicyGradient(_DIRECTIONS, _CONFIG['train'])
    p_conns, c_conns = [], []
    ## init workers

    # for _ in range(_Worker):
    #    worker.append(A2C_Worker(_CONFIG))

    for _ in range(_Worker):
        parent_conn, child_conn = Pipe()
        p_conns.append(parent_conn)
        c_conns.append(child_conn)
        workers_process.append(Process(target=worker_fkt, args=(child_conn, a2c, _CONFIG)))
    ##runner forloop
        # reset
        for worker in workers:
            worker.reset
        # send
        for conn in p_conns:
            print(conn.recv())
        # step
        for worker in workers:
            worker.start()
        # join
        for worker in workers:
            worker.join()
        # receive
        for conn in p_conns:
            print(conn.recv())


    print("finished")


if __name__ == "__main__":
    app.run(main)
