""" Data collector crypto currency data from CoinMarketCap """
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_downloader import DataDownloader
from data_preprocessor import DataPreprocessor
from model_handler import ModelHandler
from plotter import Plotter
from utils import clear_and_show_title, ensure_folders_exist

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

asset_type = "stock"
placeholder_symbol = "MAT"
validation_size = 0.2
n_days = 90


def split_data(data, frac=0.8):
    """Splits the data into training and testing data"""
    train_size = int(len(data) * frac)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data


@dataclass
class IncorrectAssetTypeError(Exception):
    """Raised when the data type entered is incorrect"""

    asset_type: str

    def __str__(self) -> str:
        return f"{self.asset_type} is an incorrect asset type; An asset is either a 'stock' or 'crypto'."


class DataCollator:
    """Data collector crypto currency data from CoinMarketCap"""

    overwrite = True
    valid_asset_types = ["crypto", "stock"]
    data = pd.DataFrame()

    def __init__(self, asset_type: str, symbol: str):
        """Initializes the DataCollector class
        Parameters
        ----------
        crypto : str
            Name of the crypto currency
        """
        if asset_type not in self.valid_asset_types:
            raise IncorrectAssetTypeError(asset_type)
        self.asset_type = asset_type
        self.symbol = symbol
        self.file_path = f"data/{symbol}.csv"
        self.downloader = DataDownloader(asset_type, placeholder_symbol, self.file_path)

    def collate_data(self):
        # Download the data
        data_exists = os.path.exists(self.file_path)
        if not data_exists or self.overwrite:
            self.downloader.download_data()
        self.data = pd.read_csv(self.file_path, index_col="Date")
        # Preprocess and scale the data using DataPreprocessor
        preprocessor = DataPreprocessor(asset_type, self.data)
        self.data = preprocessor.get_data()

    def get_data(self) -> pd.DataFrame:
        self.collate_data()
        # Preprocess data
        return self.data


def check_and_create_model(train_data, handler, validation_size):
    model = handler.load_model()
    if model is None:
        # TODO: Create train and validation data
        x_train, x_val, y_train, y_val = train_test_split(
            train_data, train_data, test_size=validation_size, shuffle=False
        )
        # Create and train model
        model, _ = handler.create_model(x_train, y_train, x_val, y_val)
        handler.save_model()
        print("[INFO]\tModel created and saved.")
    else:
        print("[INFO]\tModel loaded from storage.")
    return model


def main():
    """Main function"""

    clear_and_show_title()
    # Download and get data
    collector = DataCollator(asset_type, placeholder_symbol)
    data = collector.get_data()
    # Print phase 1 data to console
    print("[INFO]\tPhase 1 data")
    print(data.tail())

    # Preprocesses the data
    preprocessor = DataPreprocessor(asset_type, data)
    data = preprocessor.scale_data(data)
    # Print phase 2 data to console
    print("[INFO]\tPhase 2 data")
    print(data.tail())
    
    # Split the dataframe into training and testing data
    train_data, test_data = split_data(data)

    # Perform next step after getting train_data from dataframe
    train_close_prices = train_data["Close"].to_numpy()

    handler = ModelHandler()
    model = check_and_create_model(handler, train_close_prices, validation_size, n_days)
    test_close_prices = test_data["Close"].to_numpy()

    # x_test = prepare_testing_data(data, n_days)
    predicted_prices = handler.test_model(model, test_close_prices)
    rmse = np.sqrt(mean_squared_error(test_close_prices, predicted_prices))
    print("Root Mean Squared Error:", rmse)
    # Create a dataframe containing two columns. One column with price and the other column stating whether the price is actual or predicted then merge into a single dataframe
    # Filter test data to last n days
    test_data = test_data[-30:]
    # Create Date column
    test_data["Date"] = test_data.index

    # Create a new column indicating whether the price is actual or predicted
    test_data["Type"] = "Actual"
    # Get latest date from test data index
    today = test_data.index[-1]
    # get the next day in dd/mm/yyyy format
    today = datetime.strptime(today, "%d/%m/%Y") + timedelta(days=1)
    future_day = today.strftime("%d/%m/%Y")
    # Add predicted prices as new row to test data for next day (i.e. the day after the last day of the test data)
    test_data = test_data.append(
        {
            "Date": future_day,
            "Close": predicted_prices[0][0],
            "Type": "Predicted",
        },
        ignore_index=True,
    )
    test_data = preprocessor.rescale(test_data)
    test_data.to_csv("results/test_data.csv", index=False)
    Plotter.plot_results(test_data)

    print("[INFO]\tProgram finished successfully.")


if __name__ == "__main__":
    ensure_folders_exist()
    st = time.time()
    main()
    et = time.time()
    print(f"[INFO]\tTime taken: {et-st:.2f} seconds")
