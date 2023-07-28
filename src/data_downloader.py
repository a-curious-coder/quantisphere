import pandas as pd
import yfinance as yf
from cryptocmd import CmcScraper


class DataDownloader:
    def __init__(self, asset_type: str, symbol: str, filepath: str):
        """Initializes the DataCollector class
        Parameters
        ----------
        crypto : str
            Name of the crypto currency
        """
        self.asset_type = asset_type
        self.symbol = symbol
        self.file_path = filepath

    def download_crypto_data(self) -> pd.DataFrame:
        """Download stock data using yfinance library"""
        scraper = CmcScraper(self.symbol, fiat="GBP")
        data = scraper.get_dataframe()
        data.to_csv(self.file_path)

    def download_stock_data(self):
        """Download stock data using yfinance library"""
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period="max")
        data.index = data.index.strftime("%d/%m/%Y")

        data.to_csv(self.file_path)

    def download_data(self):
        if self.asset_type == "crypto":
            self.download_crypto_data()
        elif self.asset_type == "stock":
            self.download_stock_data()
