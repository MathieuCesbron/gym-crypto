import pandas as pd

df = pd.read_csv('data/BTCUSDT.csv')
aya = df.loc[df['high'].idxmax()]['high']
print(aya)