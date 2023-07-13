""" Data collector crypto currency data from CoinMarketCap """
import os
import numpy as np
import pandas as pd
import yfinance as yf
from cryptocmd import CmcScraper
from data_preprocessor import Preprocessor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
# Import go
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.config.optimizer.set_jit(True)

import joblib

def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except FileNotFoundError:
        return None

def save_model(model):
    joblib.dump(model, 'model.pkl')

def load_data(stock):
    data_collector = DataCollector(stock=stock)
    data = data_collector.get_stock_data()
    return data

def split_data(data):
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    return train_data, test_data

def scale_data(train_data):
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))
    return scaler, scaled_train_data

def create_input_target(train_data, n_days):
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
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
    return model, history

def prepare_testing_data(data, scaler, n_days):
    inputs = data['Close'].values[-n_days:]
    inputs = scaler.transform(inputs.reshape(-1, 1))
    x_test = []
    for i in range(n_days, len(inputs) + 1):  # Include the equal sign
        x_test.append(inputs[i-n_days:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test


def make_predictions(model, x_test, scaler):
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
            processed_data = pd.read_csv(f"data/{self.crypto}.csv", index_col=0)
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
            processed_data = pd.read_csv(f"data/{self.stock}.csv", index_col=0)
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
    n_days = 90
    model = load_model()
    data = load_data(stock="MAT")
    train_data, test_data = split_data(data)
    scaler, scaled_train_data = scale_data(train_data)
    x_train, y_train = create_input_target(scaled_train_data, n_days)
    x_train = reshape_input(x_train)
    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
    if model is None:
        model, history = create_model(x_train, y_train, x_val, y_val)  # Modify this line
        save_model(model)
    else:
        print("Model loaded from storage.")

    x_test = prepare_testing_data(data, scaler, n_days)
    predicted_prices = make_predictions(model, x_test, scaler)

    # Filter test data to last n days
    test_data = test_data[-n_days:]
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=test_data.index, y=predicted_prices, mode='markers', name='Predicted Price', marker=dict(color='green', size=30, symbol='x')))


    fig.update_traces(marker=dict(color='green', size=10, symbol='circle'), selector=dict(name='Predicted Price'))
    fig.update_traces(line=dict(color='yellow', width=1, dash='dot'), selector=dict(name='_line'))

    fig.update_yaxes(range=[min(test_data['Close']) * 0.8, max(test_data['Close']) * 1.2])
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickvals=fig.layout.yaxis.tickvals, tickmode='array', ticktext=fig.layout.yaxis.ticktext, tickangle=0)
    fig.update_xaxes(range=[test_data.index[0], test_data.index[-1]])
    fig.update_layout(template='plotly_dark')
    fig.update_layout(height=1080, width=1920)
    fig.update_layout(title=f"MAT Price Prediction", xaxis_title='Time', yaxis_title='Price')

    import plotly.io as pio
    # Save the plot as png
    pio.write_image(fig, "images/MAT2.png")

    # Evaluate the model
    # evaluate_model(test_data, predicted_prices)


if __name__ == "__main__":
    main()
