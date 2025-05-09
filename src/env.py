import gymnasium as gym
from ale_py import ALEInterface
from src.utils import discretize_state

class SpaceInvadersEnv:
    def __init__(self, bins=10):
        """
        Start de SpaceInvaders-omgeving en stelt het aantal bins in.
        
        Parameters:
            bins: Het aantal bins voor elke dimensie van de state.
        """
        self.env = gym.make("ALE/SpaceInvaders-v5")

        self.bins = bins

    def reset(self):
        state, info = self.env.reset()
        state = discretize_state(state, self.bins)
        return state

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        next_state = discretize_state(next_state, self.bins)
        return next_state, reward, done, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space