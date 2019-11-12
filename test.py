import pandas as pd
import numpy as np

df = pd.read_csv('data/BTCUSDT.csv')

frame = np.array([df.loc[0:5, 'open'].values, df.loc[0:5, 'open'].values])
print(frame)
