
# Q-Learning en Random Agent voor CartPole

Dit project bevat een implementatie van twee verschillende agenten voor de CartPole omgeving in Gymnasium: een **Q-Learning agent** en een **Random agent**. De agents worden getest door middel van simulaties en hun prestaties worden vergeleken.

## Installatie

Om dit project te draaien, heb je de volgende Python-pakketten nodig:

```bash
pip install -r requirements.txt
````

## Gebruik

1. **Train een Random Agent**
   Het trainen van de Random agent gebeurt door het uitvoeren van:

   ```python
   random_rewards, random_agent = train_agent(n_episodes=150, bins=10, use_random=True)
   ```

2. **Train een Q-Learning Agent**
   Het trainen van de Q-Learning agent gebeurt door het uitvoeren van:

   ```python
   qlearning_rewards, qlearning_agent = train_agent(n_episodes=150, bins=10, use_random=False)
   ```

3. **Vergelijk de prestaties**
   Je kunt de beloningen (rewards) van beide agents vergelijken door de `random_rewards` en `qlearning_rewards` variabelen te plotten.

## Bestanden

* `agents.py`: Bevat de implementatie van de `QLearningAgent` en `RandomAgent` klassen.
* `train_agent.py`: Bevat de logica voor het trainen van de agents en het uitvoeren van simulaties.
* `CartPole-v1`: De Gymnasium omgeving die gebruikt wordt voor de experimenten.

## Auteur

Dit project is gemaakt door Nima, Tommi, Vince en Isa

