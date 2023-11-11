import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})

# Hyperparameters
episodes = 1000  # Total number of episodes
alpha = 0.5  # Learning rate
gamma = 0.9  # Discount factor

outcomes = []

# frozen lake environment gets initialised
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)

nb_states = env.observation_space.n
nb_actions = env.action_space.n

# Q-Table gets created with the correct size
qtable = np.zeros((nb_states, nb_actions))

# main loop
for i in range(episodes):
    state, prob = env.reset()
    done = False

    outcomes.append("Failure")

    # while the current episode is not finished
    while not done:
        # if the Q-Table has a value on this state, take the best action
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])

        # if the Q-Table is empty on this state, take a random action
        else:
            action = env.action_space.sample()

        # get step information
        new_state, reward, done, info, prob = env.step(action)

        # calculate the new Q-Value
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        # update the current state
        state = new_state

        # if we get a reward on this episode, save it as such
        if reward:
            outcomes[-1] = "Success"

print()
print('===========================================')
print('Q-table after training:')
print(qtable)

# Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()