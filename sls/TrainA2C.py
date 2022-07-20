import time

from absl import app
from sls import Env, Runner
from sls.agents import *
from tensorflow import keras
from sls.learn import A2C_PolicyGradient
from multiprocessing import Process, Pipe
from sls.worker import A2C_Worker

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
episode = 0

def main(unused_argv):
    workers = []
    workers_process = []
    a2c = A2C_PolicyGradient(_CONFIG['train'])
    p_conns, c_conns = [], []
    ## init workers

    # for _ in range(_Worker):
    #    worker.append(A2C_Worker(_CONFIG))

    for id in range(_Worker):
        parent_conn, child_conn = Pipe()
        p_conns.append(parent_conn)
        c_conns.append(child_conn)
        workers_process.append(A2C_Worker(id, _CONFIG, child_conn))
    print(len(p_conns), len(c_conns), 'Connections')
    for p in workers_process:
        p.start()
    for in_conn in p_conns:
        print('receive_start')
        in_conn.recv()
    while episode <= _CONFIG['episodes']:
        for out_conn in p_conns:
            out_conn.send(["RESET"])
        for in_conn in p_conns:
            print('receive_reset')
            in_conn.recv()
        while True:
            for out_conn in p_conns:
                out_conn.send(["STEP"])
            worker_done = []
            for in_conn in p_conns:
                print('waiting for rec', in_conn)
                sar_batch, done, last = in_conn.recv()
                worker_done.append(last)
                if _CONFIG['train'] and sar_batch is not None:
                    if done:
                        a2c.add_last_to_batch(sar_batch)
                    else:
                        a2c.add_to_batch(sar_batch)
            if any(worker_done): #all worker should finish at the same time
                break

#         self.score += obs.reward
#        self.summarize()
    print("finished")


if __name__ == "__main__":
    app.run(main)
