import datetime
import pandas as pd
from time import sleep
from binance.client import Client
import keys


def data_creation(apikey, secretkey, pair, since):
    CLIENT = Client(keys.apiKey, keys.secretKey)

    DATA_TO_CSV = CLIENT.get_historical_klines(pair,
                                               Client.KLINE_INTERVAL_1MINUTE,
                                               since)

    DATA_TO_CSV = pd.DataFrame(
        DATA_TO_CSV,
        columns=[
            "Open time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close time",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "Ignore",
        ],
    )
    DATA_TO_CSV.set_index("Open time", inplace=True)
    DATA_TO_CSV.to_csv(pair + ".csv")


def stationarization(csv):
    output = pd.DataFrame()
    return output


data_creation(keys.apiKey, keys.secretKey, "ETHUSDT", "1 hour ago UTC")
