import matplotlib.pyplot as plt
from ale_py import ALEInterface
from src.env import SpaceInvadersEnv
from src.agents import QLearningAgent, RandomAgent

def train_agent(n_episodes=500, bins=10, use_random=False):
    """
    Train de RL-agent op de SpaceInvaders-omgeving.
    
    Parameters:
        n_episodes: Aantal episodes om te trainen.
        bins: Aantal bins voor discretisatie van de state.
        use_random: Indien True wordt de RandomAgent gebruikt als baseline.
        
    Returns:
        Een tuple bestaande uit (rewards_per_episode, agent)
    """
    env = SpaceInvadersEnv(bins=bins)
    if use_random:
        agent = RandomAgent(env.action_space)
    else:
        agent = QLearningAgent(env.action_space, state_bins=bins)
    
    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if not use_random:
                agent.update(state, action, reward, next_state, done)
            state = next_state
        if not use_random:
            agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon if not use_random else 'N/A'}")
    
    env.close()
    return rewards_per_episode, agent

if __name__ == "__main__":
    episodes = 500
    rewards, agent = train_agent(n_episodes=episodes, bins=10, use_random=False)
    
    # Visualiseer de resultaten: total reward per episode
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward per Episode")
    plt.show()