import gymnasium as gym
from ale_py import ALEInterface
from src.utils import discretize_state

class SpaceInvadersEnv:
    def __init__(self, bins=10):
        """
        Initialiseert de SpaceInvaders-omgeving (RAM-versie) en stelt het aantal bins in
        voor de discretisering van de state.
        
        Parameters:
            bins: Het aantal bins voor elke dimensie van de state.
        """
        self.env = gym.make("ALE/SpaceInvaders-v5")
        self.bins = bins

    def reset(self):
        """
        Resetten van de omgeving en direct discretiseren van de state.
        """
        state, info = self.env.reset()
        state = discretize_state(state, self.bins)
        return state

    def step(self, action):
        """
        Voert een actie uit in de omgeving en discretiseert de volgende state.
        """
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