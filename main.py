from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env import CryptoEnv
import pandas as pd
import os

df = pd.read_csv('data/BTCUSDT.csv', index_col=0)

env = DummyVecEnv([lambda: CryptoEnv(df)])

# Instanciate the agent
model = PPO2(MlpPolicy, env, gamma=1, learning_rate=0.01, verbose=0)

# Train the agent
total_timesteps = int(os.getenv('TOTAL_TIMESTEPS', 500000))
model.learn(total_timesteps)

# Render the graph of rewards
env.render(graph=True)

# Save the agent
# model.save('PPO2_CRYPTO')

# Trained agent performence
obs = env.reset()
env.render()
for i in range(100000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(print_step=True)
