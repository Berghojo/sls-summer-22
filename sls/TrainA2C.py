import os
import datetime
from absl import app
from sls.agents import *
from tensorflow import keras
from sls.learn import A2C_PolicyGradient
from multiprocessing import Process, Pipe
from sls.worker import A2C_Worker
import tensorflow as tf
import numpy as np

_CONFIG = dict(
    episodes=10000,
    screen_size=16,
    minimap_size=64,
    visualize=True,
    train=True,
    agent=A2C_Agent,
    load_path='./models/abgabe05_aufgabe01_model_weights.h5'
)

path = './graphs/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
       + ('_train_' if _CONFIG['train'] else 'run_') \
       + type(_CONFIG['agent']).__name__
# Tensorflow 1.X
# self.writer = tf.summary.FileWriter(self.path, tf.get_default_graph())
# Tensorflow 2.X mit ausgeschalteter eager_execution
# Alle weiteren tf.summary Aufrufe müssen durch tf.compat.v1.summary tf.compat.v1.summary ersetzt werden
writer = tf.compat.v1.summary.FileWriter(path, tf.compat.v1.get_default_graph())
ospath = os.path.isfile(_CONFIG['load_path'])

_Worker = 8
episode = 1
score_batch = [0] * _Worker
worker_done = []

def main(unused_argv):
    workers = []
    global worker_done
    global episode
    global score_batch
    workers_process = []
    a2c = A2C_PolicyGradient(_CONFIG['train'])
    if not _CONFIG['train'] and _CONFIG['load_path'] is not None and os.path.isfile(_CONFIG['load_path']):
        a2c.load_model_weights(_CONFIG['load_path'])
    p_conns, c_conns = [], []
    ## init workers

    for id in range(_Worker):
        parent_conn, child_conn = Pipe()
        p_conns.append(parent_conn)
        c_conns.append(child_conn)
        workers_process.append(A2C_Worker(id, _CONFIG, child_conn))
    for p in workers_process:
        p.start()
    for in_conn in p_conns:
        in_conn.recv()
    for out_conn in p_conns:
        out_conn.send(["RESET"])
    # read empty scores
    for in_conn in p_conns:
        in_conn.recv()
    # wait for Worker RDY
    for in_conn in p_conns:
        in_conn.recv()
    while episode <= _CONFIG['episodes']:
        while True:
            worker_done = []
            for out_conn in p_conns:
                out_conn.send(["STEP"])
            states = [0] * _Worker
            skip = False
            for in_conn in p_conns:
                return_s = in_conn.recv()
                if return_s == [None, None]:
                    skip = True
                    continue
                worker_id, state = return_s
                states[worker_id] = state
            if not skip:
                actions, values = a2c.choose_action(states)
                for i, in_conn in enumerate(p_conns):
                    in_conn.send([actions[i], values[i]])

            for in_conn in p_conns:
                recv_2 = in_conn.recv()
                sar_batch, done, last = recv_2
                worker_done.append(last)
                if _CONFIG['train'] and sar_batch is not None:
                    if done:
                        a2c.add_last_to_batch(sar_batch)
                    else:
                        a2c.add_to_batch(sar_batch)

            if all(worker_done):  # all worker should finish at the same time
                summarize(a2c)
                episode += 1
                score_batch = []
                if episode <= _CONFIG['episodes']:
                    for out_conn in p_conns:
                        out_conn.send(["RESET"])
                else:
                    for out_conn in p_conns:
                        out_conn.send(["CLOSE"])
                for in_conn in p_conns:
                    score_batch.append(in_conn.recv())
                for in_conn in p_conns:
                    in_conn.recv()
                break
    for p in workers_process:
        p.kill()
    print("killed")

def summarize(a2c):
    global episode
    mean = np.mean(score_batch)
    writer.add_summary(tf.compat.v1.Summary(
        value=[tf.compat.v1.Summary.Value(tag='Average Worker score', simple_value=mean)]),
        episode)

    print('Mean Score: ', mean, 'in Episode:', episode)
    if _CONFIG['train'] and episode % 10 == 0:
        a2c.save_model_weights()


if __name__ == "__main__":
    app.run(main)
