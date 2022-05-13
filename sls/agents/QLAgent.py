from sls.agents import AbstractAgent
from sls.learn import QLTable


class QLAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(QLAgent, self).__init__(screen_size)
        self.state_space = 2
        self.index_values = int(screen_size / self.state_space)
        self.train = train
        self.last_state = None
        self.last_action = None
        self.qtable = QLTable(self._DIRECTIONS, screen_size, 0.0)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            beacon = self._get_beacon(obs)
            marine_coords = self._get_unit_pos(marine) # walks in 3px steps
            beacon_coords = self._get_unit_pos(beacon)
            if marine is None:
                return self._NO_OP

            state = self.get_state(marine_coords, beacon_coords, obs)
            direction = self.qtable.choose_action(state)

            if self.last_state:
                self.qtable.learn(self.last_state, self.last_action, state, obs)

            if state == 'target':
                self.last_action = None
                self.last_state = None
            else:
                self.last_action = direction
                self.last_state = state

            return self._dir_to_sc2_action(direction, marine_coords)
        else:
            return self._SELECT_ARMY

    def get_state(self, marine_coords, beacon_coords, obs):
        distance = beacon_coords - marine_coords
        state = (distance / self.qtable.state_space).astype(int)
        state_string = f'{state[0]} {state[1]}'
        if obs.reward != 0:
            state_string = 'target'

        #if state_string == self.last_state:
        #    print('no transition')
        return state_string

    def save_model(self, path):
        self.qtable.save_qtable(path)

    def load_model(self, filename):
        self.qtable.load_qtable(filename)

    def update_epsilon(self, episodes):
        epsilon = self.qtable.epsilon - (1 / (episodes - 100))
        if epsilon < 0.001:
            epsilon = 0.0
        #update epsilon
        self.qtable.epsilon = epsilon

    def get_epsilon(self):
        return self.qtable.epsilon