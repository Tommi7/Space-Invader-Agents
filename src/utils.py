# utils.py
import numpy as np

def discretize_state(state, bins):
    """
    Discretiseert een continue state door elk element in de state te kwantiseren.
    Deze functie gaat ervan uit dat de state-waarden in het bereik [0, 255] liggen,
    wat typisch is voor de RAM-versie van Atari-omgevingen.
    
    Parameters:
        state: De continue state als een numpy-array of lijst.
        bins: Het aantal bins om elke dimensie in te delen.
        
    Returns:
        Een tuple die de gediscretiseerde state voorstelt.
    """
    state = np.array(state).flatten()  # Zorg dat alle elementen in één dimensie komen
    # Normaliseer naar het interval [0, 1]
    state_normalized = state / 255.0
    discretized = np.floor(state_normalized * bins).astype(int)
    # Converteer de array naar een gewone tuple van ints (geen geneste lijsten)
    return tuple(discretized.tolist())