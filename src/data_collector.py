""" Data collector crypto currency data from CoinMarketCap """
from dataclasses import dataclass
import os
import warnings
from datetime import datetime, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from cryptocmd import CmcScraper
from pyautogui import size as pyautogui_size
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_preprocessor import DataPreprocessor
from utils import clear_and_show_title, ensure_folders_exist

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

asset_type = "stock"
placeholder_symbol = "MAT"
validation_size = 0.2
train_n_days = 90

@dataclass
class IncorrectModelTypeError(Exception):
    """ Raised when the data type entered is incorrect """
    model_type: str
    
    def __str__(self) -> str:
        return f"{self.model_type} is an incorrect model type; A model is either a 'LSTM' or 'GRU'."


class ModelHandler:
    model_name = 'model.pkl'
    valid_model_types = ["LSTM", "GRU"]

    def __init__(self, data: pd.DataFrame, model_type: str = "LSTM"):
        if model_type not in valid_model_types:
            raise IncorrectModelTypeError(model_type)
        self.model_type = model_type

    def create_model(x_train, y_train, x_val, y_val):
        """ Creates and trains the model """
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
        return model, history

    def test_model(self, model, x_test):
        # Makes prediction and restores the original scale
        predicted_prices = model.predict(x_test)
        return predicted_prices

    def evaluate_model(self, test_data, predicted_prices):
        mse = mean_squared_error(test_data['Close'], predicted_prices)
        print("Mean Squared Error:", mse)
        
    def load_model(self) -> object:
        """ Loads the model """
        try:
            self.model = joblib.load(self.model_name)
        except FileNotFoundError:
            return None
        return self.model

    def save_model(self):
        """ Saves the model """
        joblib.dump(self.model, self.model_name)

def preprocess_data(data) -> tuple[MinMaxScaler, pd.DataFrame]:
    """ Preprocesses all data """
    preprocessor = DataPreprocessor()
    scaler, data = preprocessor.scale_data(data)
    # Transform Date column to index
    data.set_index('Date', inplace=True)
    # Filter data to only include the Date and Close column
    data = data[['Close']]
    return (scaler, data)

def split_data(data, train_size = 0.8):
    """ Splits the data into training and testing data """
    train_size = int(len(data) * train_size)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    return train_data, test_data

def create_input_target(train_data, n_days):
    """ Creates the input and target data 
    NOTE: n_days is used as the pattern to forecast the n_days+1 day price
    """
    x_train = []
    y_train = []
    for i in range(n_days, len(train_data)):
        x_train.append(train_data[i-n_days:i, 0])
        y_train.append(train_data[i, 0])
    return np.array(x_train), np.array(y_train)

def reshape_input(x_train):
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train

def prepare_testing_data(data, n_days):
    """ Prepares the testing data """
    inputs = data['Close'].values[-n_days:]
    x_test = []
    for i in range(n_days, len(inputs) + 1):  # Include the equal sign
        x_test.append(inputs[i-n_days:i])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test

def plot_results(test_data, predicted_prices):
    plt.plot(test_data['Close'].values, color='black', label='Actual Price')
    plt.plot(predicted_prices, color='green', label='Predicted Price')
    plt.title(f"{placeholder_symbol} Price Prediction")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()



@dataclass
class IncorrectAssetTypeError(Exception):
    """ Raised when the data type entered is incorrect """
    asset_type: str
    
    def __str__(self) -> str:
        return f"{self.asset_type} is an incorrect asset type; An asset is either a 'stock' or 'crypto'."

class DataCollector:
    """ Data collector crypto currency data from CoinMarketCap """
    valid_asset_types = ["crypto", "stock"]
    def __init__(self, asset_type: str, symbol: str):
        """ Initializes the DataCollector class
        Parameters
        ----------
        crypto : str
            Name of the crypto currency
        """
        if asset_type not in self.valid_asset_types:
            raise IncorrectAssetTypeError(asset_type)
        self.symbol = symbol
        self.file_path = f"data/{symbol}.csv"

    def download_crypto_data(self) -> pd.DataFrame:
        """ Download stock data using yfinance library """
        scraper = CmcScraper(self.crypto, start_date=self.start_date, end_date=self.end_date, fiat="GBP")
        data = scraper.get_dataframe()
        processed_data = DataPreprocessor.crypto(data)
        self.save(processed_data)

    def download_stock_data(self):
        """ Download stock data using yfinance library """
        ticker = yf.Ticker(self.stock)
        data = ticker.history(period="max")
        processed_data = DataPreprocessor.stocks(data)
        self.save(processed_data)

    def download_data(self):
        if(self.asset_type == "crypto"):
            self.download_crypto_data()
        elif(self.asset_type == "stock"):
            self.download_stock_data()

    def get_data(self) -> pd.DataFrame:
        data_exists = os.path.exists(self.file_path)
        if not data_exists:
            self.download_data()

        data = pd.read_csv(self.file_path)

        return data

    def save(self, data):
        """ Save the preprocessed stock data to a file """
        data.to_csv(self.file_path)


def main():
    """ Main function """
    clear_and_show_title()
    collector = DataCollector(asset_type, placeholder_symbol)
    data = collector.get_data()
    
    preprocessor = DataPreprocessor(asset_type, data)
    preprocessor.run()
    data = preprocessor.get_data()
    
    train_data, test_data = split_data(data)
    
    handler = ModelHandler(data)
    model = handler.load_model()
    

    if model is None:
        x_train, y_train = create_input_target(train_data, train_n_days)
        x_train = reshape_input(x_train)
        # NOTE: We will use the last 20% of the training data as validation data
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, shuffle=False)
        model, _ = handler.create_model(x_train, y_train, x_val, y_val)  # Modify this line
        handler.save_model(model)
    else:
        print("[INFO]\tModel loaded from storage.")

    x_test = prepare_testing_data(data, train_n_days)
    predicted_prices = handler.test_model(model, x_test)
    # Create a dataframe containing two columns. One column with price and the other column stating whether the price is actual or predicted then merge into a single dataframe
    # Filter test data to last n days
    test_data = test_data[-30:]
    # Create a new column indicating whether the price is actual or predicted
    test_data['Type'] = 'Actual'
    
    # Get latest date from test data
    today = test_data['Date'].iloc[-1]
    # get the next day in dd/mm/yyyy format
    today = datetime.strptime(today, '%d/%m/%Y') + timedelta(days=1)
    future_day = today.strftime('%d/%m/%Y')
    
    # Add predicted prices as new row to test data for next day (i.e. the day after the last day of the test data)
    test_data = test_data.append({'Date': future_day, 'Close': predicted_prices[0][0], 'Type': 'Predicted'}, ignore_index=True)
    test_data.to_csv("results/test_data.csv", index=False)
    test_data = preprocessor.rescale(test_data)
    print(test_data.tail())


    # Assuming you have the data in a pandas DataFrame called 'data'
    fig = go.Figure()

    # filter data to only include actual closing price and predicted closing price
    predicted_data = test_data[test_data['Type'] != 'Actual']
    actual_data = test_data[test_data['Type'] != 'Predicted']
    # Add the actual closing price
    fig.add_trace(go.Scatter(x=actual_data['Date'], y=actual_data['Close'], name='Actual Closing Price'))

    # thickens the lines and change marker size
    fig.update_traces(line=dict(width=2), marker=dict(size=5))
    
    # Add the predicted closing price
    fig.add_trace(go.Scatter(x=predicted_data['Date'], y=predicted_data['Close'], name='Predicted Closing Price'))

    # Get this screen size
    screen_size = pyautogui_size()
    # Customize the layout
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Closing Price',
        template='plotly_dark',
        # size is this screen size
        width=screen_size[0],
        height=screen_size[1]
    )
    # change marker symbol and colour and size
    fig.update_traces(mode='markers+lines', marker=dict(symbol='diamond-open', color='white', size=10))
    # Save plot to png
    pio.write_image(fig, "images/MAT.png")


    # Evaluate the model
    # evaluate_model(test_data, predicted_prices)


if __name__ == "__main__":
    ensure_folders_exist()
    main()
