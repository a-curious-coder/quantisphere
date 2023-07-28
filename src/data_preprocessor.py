""" This module preprocesses the data """
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    """Preprocesses the data"""

    scaler = MinMaxScaler()

    def __init__(self, asset_type: str, data: pd.DataFrame) -> None:
        self.asset_type = asset_type
        self.data = data
        self.preprocessers = {
            "stock": self.preprocess_stock_data,
            "crypto": self.preprocess_crypto_data,
        }

    def preprocess(self):
        """Runs the data preprocessor"""
        try:
            choice = self.preprocessers.get(self.asset_type, self.invalid_asset_type)
            choice()
        except Exception as e:
            raise ValueError("Invalid asset type")

    def preprocess_crypto_data(self):
        """Preprocesses the crypto data"""
        # Remove Market Cap column
        self.data.drop(columns=["Market Cap"], inplace=True)
        # Convert Date column to string
        self.data["Date"] = self.data["Date"].astype(str)
        # Reformat date from yyyy-mm-dd to dd-mm-yyyy
        self.data["Date"] = self.data["Date"].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%d/%m/%Y")
        )

        self.data["Close"] = self.data["Close"].apply(
            lambda x: round(float(x), 2) if float(x) > 10 else float(x)
        )

    def preprocess_stock_data(self):
        """Preprocesses the stock data"""
        # Filter data to only include the Date and Close column
        self.data = self.data[["Close"]]

        self.data["Close"] = self.data["Close"].apply(
            lambda x: round(float(x), 2) if float(x) > 10 else float(x)
        )

    def invalid_asset_type(self):
        """Raises an error if the asset type is invalid"""
        raise ValueError("Invalid asset type")

    def scale_data(self, data):
        """Scales the data"""
        data["Close"] = self.scaler.fit_transform(data["Close"].values.reshape(-1, 1))
        return data

    def rescale(self, data):
        """Rescales the data"""
        data["Close"] = self.scaler.inverse_transform(
            data["Close"].values.reshape(-1, 1)
        )
        return data

    def get_data(self):
        """Returns the scaled data"""
        self.preprocess()
        return self.data
