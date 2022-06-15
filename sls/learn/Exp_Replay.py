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
