from stable_baselines.ddpg.policies import CnnPolicy, LnCnnPolicy, MlpPolicy, LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG

from env import CryptoEnv
import pandas as pd
import os

policies = [MlpPolicy, LnMlpPolicy]
df = pd.read_csv('data/BTCUSDT.csv', index_col=0)
one_day_data = df.loc[1000:2440]

max_timesteps = 10000

env = DummyVecEnv([lambda: CryptoEnv(df)])

for pi in policies:

    # Instanciate the agent
    model = DDPG(pi, env)

    # Train the agent
    model.learn(max_timesteps)

    # Render the graph of rewards
    env.render(graph_reward=True)

    # Trained agent performance
    obs = env.reset()
    env.render()
    for i in range(len(df)):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(print_step=True)