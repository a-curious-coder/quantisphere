""" Data collector crypto currency data from CoinMarketCap """
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

from data_preprocessor import Preprocessor
from utils import clear_and_show_title, ensure_folders_exist

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_model():
    """ Loads the model """
    try:
        model = joblib.load('model.pkl')
        return model
    except FileNotFoundError:
        return None

def save_model(model):
    """ Saves the model """
    joblib.dump(model, 'model.pkl')

def load_data(stock):
    """ Loads the data """
    data_collector = DataCollector(stock=stock)
    data = data_collector.get_stock_data()
    return data

def preprocess_data(data):
    """ Preprocesses all data """
    preprocessor = Preprocessor()
    scaler, data = preprocessor.scale_data(data)
    # Transform Date column to index
    data.set_index('Date', inplace=True)
    # Filter data to only include the Date and Close column
    data = data[['Close']]
    return scaler, data

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

def prepare_testing_data(data, n_days):
    """ Prepares the testing data """
    inputs = data['Close'].values[-n_days:]
    x_test = []
    for i in range(n_days, len(inputs) + 1):  # Include the equal sign
        x_test.append(inputs[i-n_days:i])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test


def make_predictions(model, x_test, scaler):
    # Makes prediction and restores the original scale
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

def plot_results(test_data, predicted_prices):
    plt.plot(test_data['Close'].values, color='black', label='Actual Price')
    plt.plot(predicted_prices, color='green', label='Predicted Price')
    plt.title(f"{data_collector.symbol} Price Prediction")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def evaluate_model(test_data, predicted_prices):
    mse = mean_squared_error(test_data['Close'], predicted_prices)
    print("Mean Squared Error:", mse)

class DataCollector:
    """ Data collector crypto currency data from CoinMarketCap """
    def __init__(self, crypto=None, stock=None, start_date = None, end_date = None):
        """ Initializes the DataCollector class
        Parameters
        ----------
        crypto : str
            Name of the crypto currency
        """
        self.crypto = crypto
        self.stock = stock
        self.symbol = crypto if crypto is not None else stock
        self.start_date = start_date
        self.end_date = end_date
        if start_date is not None and end_date is not None:
            self.file_path = f"data/{crypto}_{start_date}_{end_date}.csv"
        else:
            self.file_path = f"data/{crypto}.csv"

    def download_crypto_data(self):
        """ Download stock data using yfinance library """
        scraper = CmcScraper(self.crypto, start_date=self.start_date, end_date=self.end_date, fiat="GBP")
        data = scraper.get_dataframe()
        return data

    def download_stock_data(self):
        """ Download stock data using yfinance library """
        ticker = yf.Ticker(self.stock)
        data = ticker.history(period="max")
        return data

    def get_crypto(self):
        """ Get stock data by calling the smaller functions """
        if os.path.exists(f"data/{self.crypto}.csv"):
            # If the data file already exists, load it instead of downloading
            processed_data = pd.read_csv(f"data/{self.crypto}.csv")
        else:
            # If the data file does not exist, download the data
            data = self.download_stock_data()
            processed_data = Preprocessor.crypto(data)
            self.save(processed_data)

        return processed_data

    def get_stock_data(self):
        """ Get stock data by calling the smaller functions """
        if os.path.exists(f"data/{self.stock}.csv"):
            # If the data file already exists, load it instead of downloading
            processed_data = pd.read_csv(f"data/{self.stock}.csv")
        else:
            # If the data file does not exist, download the data
            data = self.download_stock_data()
            processed_data = Preprocessor.stocks(data)
            self.save(processed_data)

        return processed_data

    def save(self, data):
        """ Save the preprocessed stock data to a file """
        data.to_csv(f"data/{self.symbol}.csv")


def main():
    """ Main function """
    clear_and_show_title()
    validation_size = 0.2
    placeholder_symbol = "MAT"
    train_n_days = 90
    model = load_model()
    data = load_data(stock="MAT")
    # NOTE: Scaled all data; we will save for rescaling later
    scaler, scaled_data = preprocess_data(data)
    train_data, test_data = split_data(scaled_data)

    if model is None:
        x_train, y_train = create_input_target(train_data, train_n_days)
        x_train = reshape_input(x_train)
        # NOTE: We will use the last 20% of the training data as validation data
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, shuffle=False)
        model, _ = create_model(x_train, y_train, x_val, y_val)  # Modify this line
        save_model(model)
    else:
        print("[INFO]\tModel loaded from storage.")

    x_test = prepare_testing_data(data, train_n_days)
    predicted_prices = make_predictions(model, x_test, scaler)
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
