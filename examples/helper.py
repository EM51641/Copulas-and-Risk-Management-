import pandas as pd  # type: ignore
from yfinance import download  # type: ignore


def get_data(tickers: list[str]) -> pd.DataFrame:
    """
    Get data from Yahoo Finance
    """
    df = pd.DataFrame()
    for ticker in tickers:
        df[ticker] = download(ticker, start="1998-01-01")["Close"]
    return df
