from sls.agents import AbstractAgent
from sls.learn import ExperienceReplay, DeepQNetwork



class DQN_Agent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(DQN_Agent, self).__init__(screen_size)
        self.train = train
        self.last_state = None
        self.last_action = None
        self.exp_replay = ExperienceReplay(100_000)
        self.dqn_network = DeepQNetwork()

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            beacon = self._get_beacon(obs)
            marine_coords = self._get_unit_pos(marine) # walks in 3px steps
            beacon_coords = self._get_unit_pos(beacon)
            if marine is None:
                return self._NO_OP

            state = self.get_state(marine_coords, beacon_coords)
            done = obs.reward == 1 or obs.last()
            self.exp_replay.add_experience(self.last_state, self.last_action, obs.reward, state, done)
            direction = self.dqn_network.choose_action(state)

            if self.last_state and self.train and len(self.exp_replay) > 6000:
                self.dqn_network.learn(self.exp_replay)

            if state == 'target' or obs.last():
                self.last_action = None
                self.last_state = None
            else:
                self.last_action = direction
                self.last_state = state

            return self._dir_to_sc2_action(direction, marine_coords)
        else:
            return self._SELECT_ARMY


    def get_state(self, marine_coords, beacon_coords):
        return (beacon_coords - marine_coords) / self.screen_size


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