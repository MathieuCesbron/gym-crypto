import pandas as pd
import os

df = pd.read_csv('data/BTCUSDT.csv')

MAX_QUOTE_ASSET_VOLUME = df.loc[
    df['Quote asset volume'].idxmax()]['Quote asset volume']

MAX_NUMBER_of_TRADES = df.loc[
    df['Number of trades'].idxmax()]['Number of trades']

MAX_TAKER_BUY_BASE_ASSET_VOLUME = df.loc[
    df['Taker buy base asset volume'].idxmax()]['Taker buy base asset volume']

MAX_TAKER_BUY_QUOTE_ASSET_VOLUME = df.loc[df[
    'Taker buy quote asset volume'].idxmax()]['Taker buy quote asset volume']

MAX_ACCOUNT_BALANCE = 10000000
INITIAL_ACCOUNT_BALANCE = 1000
MAX_CRYPTO_PRICE = 20000
MAX_CRYPTO = 21000000
MAKER_FEE = 0.00075
TAKER_FEE = 0.00075
BNBUSDTHELD = 1000
MAX_STEPS = 300
