import numpy as np

class QLearningAgent:
    def __init__(self, action_space, state_bins, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        """
        Q-Learning agent die een Q-table opbouwt.
        
        Parameters:
        - action_space: De action space van de omgeving (bijv. gymnasium.action_space).
        - state_bins: Het aantal bins dat gebruikt wordt voor de discretisering van de state.
        - alpha: Leersnelheid.
        - gamma: Discount-factor.
        - epsilon: Beginwaarde voor de exploratie (ε).
        - epsilon_decay: De factor waarmee ε per episode wordt verlaagd.
        - min_epsilon: De minimale waarde voor ε.
        """
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}  
        self.state_bins = state_bins

    def get_discrete_state(self, state):
        """
        Converteer de state naar een tuple zodat deze kan dienen als key in de Q-table.
        """
        return tuple(state)

    def choose_action(self, state):
        """
        Kies een actie op basis van de ε-greedy strategie.
        """
        discrete_state = self.get_discrete_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space.n)

        if np.random.rand() < self.epsilon:
            return self.action_space.sample() 
        else:
            return int(np.argmax(self.q_table[discrete_state])) 

    def update(self, state, action, reward, next_state, done):
        """
        Update de Q-waarde volgens de Q-learning update-regel.
        """
        discrete_state = self.get_discrete_state(state)
        discrete_next_state = self.get_discrete_state(next_state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space.n)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_space.n)

        max_future = np.max(self.q_table[discrete_next_state])
        current_q = self.q_table[discrete_state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * max_future

        # Q-learning update: Q(s,a) = Q(s,a) + α (target - Q(s,a))
        self.q_table[discrete_state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        """
        Verminder de exploratieparameter ε na elke episode.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, state):
        """
        Kies altijd een willekeurige actie.
        """
        return self.action_space.sample()

    def update(self, *args, **kwargs):
        """
        De RandomAgent leert niet, dus update is overbodig.
        """
        pass