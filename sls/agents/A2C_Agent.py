from sls.agents import AbstractAgent
from sls.learn import ExperienceReplay, PolicyGradient, MonteCarloEpisode
import tensorflow as tf


class A2C_Agent(AbstractAgent):

    def __init__(self, train, screen_size):
        tf.compat.v1.disable_eager_execution()
        super(PG_Agent, self).__init__(screen_size)
        self.train = train
        self.last_state = None
        self.last_action = None
        self.decay_episodes = 500
        self.episode = MonteCarloEpisode()
        self.neg_reward = -0.1
        self.pos_reward = 100
        self.policy_gradient = PolicyGradient(self._DIRECTIONS, train)

    def step(self, obs):
        if self._MOVE_SCREEN.id in obs.observation.available_actions:
            marine = self._get_marine(obs)
            beacon = self._get_beacon(obs)
            marine_coords = self._get_unit_pos(marine)
            beacon_coords = self._get_unit_pos(beacon)
            if marine is None:
                return self._NO_OP

            state = self.get_state(marine_coords, beacon_coords)
            reward = self.pos_reward if obs.reward == 1 else self.neg_reward
            done = obs.reward == 1 or obs.last()
            if self.last_state is not None and self.train:
                self.episode.add_step(self.last_state, self.last_action, reward)
            direction = self.policy_gradient.choose_action(state)

            if done and self.train:
                self.policy_gradient.learn(self.episode)
                self.last_action = None
                self.last_state = None
                self.episode.clear()
            else:
                self.last_action = direction
                self.last_state = state

            return self._dir_to_sc2_action(direction, marine_coords)
        else:
            return self._SELECT_ARMY

    def get_state(self, marine_coords, beacon_coords):
        return (beacon_coords - marine_coords) / self.screen_size

    def update_target_model(self):
        print('reset networks')
        self.policy_gradient.reset_q()

    def save_model(self, path):
        self.policy_gradient.save_model_weights(path)

    def load_model(self, filename):
        self.policy_gradient.load_model_weights(filename)

    def update_epsilon(self, episodes):
        epsilon = self.policy_gradient.epsilon - (0.95 / self.decay_episodes)
        if epsilon < 0.05:
            epsilon = 0.05
        #update epsilon
        self.policy_gradient.epsilon = epsilon

    def get_epsilon(self):
        return self.policy_gradient.epsilon
