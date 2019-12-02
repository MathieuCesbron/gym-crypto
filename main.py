from env import CryptoEnv
import pandas as pd
import os

df = pd.read_csv('data/BTCUSDT.csv', index_col=0)

env = CryptoEnv(df)

