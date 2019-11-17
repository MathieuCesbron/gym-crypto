import random
import pandas as pd

df = pd.read_csv('data/BTCUSDT.csv')

start = list(range(4, len(df.loc[:, 'Open'].values - 1)))
weights = [i**2 for i in start]

aya = random.choices(start, weights)

for i in range(100):
    print(random.choices(start, weights))
