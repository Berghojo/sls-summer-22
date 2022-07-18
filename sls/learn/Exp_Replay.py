import numpy as np


class ExperienceReplay:
    def __init__(self, size):
        self.size = size
        self.states = []
        self.actions = []
        self.rewards = []
        self.states_next = []
        self.dones = []

    def add_experience(self, state, action, reward, states_next, done):
        if len(self.states) >= self.size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.states_next.pop(0)
            self.dones.pop(0)
        self.states.append(np.array(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_next.append(states_next)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


class MonteCarloEpisode:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add_step(self, state, action, reward):
        self.states.append(np.array(state))
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def __len__(self):
        return len(self.states)


class State_Batch:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

    def add_step(self, state, action, reward, value):
        self.states.append(np.array(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)

    def pop(self, index):
        self.states.pop(index)
        self.actions.pop(index)
        self.rewards.pop(index)
        self.values.pop(index)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

    def __len__(self):
        return len(self.states)
