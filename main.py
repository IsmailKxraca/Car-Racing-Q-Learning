import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=5), map_name="4x4", render_mode="human")

env.reset()

env.render()

randomAction = env.action_space.sample()
randomStep = env.step(randomAction)

env.render()




