from env import CryptoEnv
import pandas as pd

df = pd.read_csv('data/BTCUSDT.csv')

env = CryptoEnv(df)

#la jungle
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
env = DummyVecEnv([lambda: CryptoEnv(df)])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=2000)

obs = env.reset()
env.render()
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()