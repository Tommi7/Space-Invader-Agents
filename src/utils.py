import numpy as np

def discretize_state(state, bins):
    state = np.array(state).flatten() 
    state_normalized = state / 255.0
    discretized = np.floor(state_normalized * bins).astype(int)
    return tuple(discretized.tolist())