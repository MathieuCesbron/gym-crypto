import pandas as pd

df = pd.read_csv('data/BTCUSDT.csv')

MAX_QUOTE_ASSET_VOLUME = df.loc[
    df['quote_asset_volume'].idxmax()]['quote_asset_volume']

MAX_NUMBER_of_TRADES = df.loc[
    df['number_of_trades'].idxmax()]['number_of_trades']

MAX_TAKER_BUY_BASE_ASSET_VOLUME = df.loc[
    df['taker_buy_base_asset_volume'].idxmax()]['taker_buy_base_asset_volume']

MAX_TAKER_BUY_QUOTE_ASSET_VOLUME = df.loc[df[
    'taker_buy_quote_asset_volume'].idxmax()]['taker_buy_quote_asset_volume']

MAX_ACCOUNT_BALANCE = 10000000
INITIAL_ACCOUNT_BALANCE = 1000
MAX_CRYPTO_PRICE = 20000
MAX_CRYPTO = 21000000
MAKER_FEE = 0.00075
TAKER_FEE = 0.00075
MAX_STEPS = 20000