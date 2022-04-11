#!/usr/bin/env python

import os
from os.path import exists

import numpy as np
import pandas as pd
from cryptocmd import CmcScraper
from currency_converter import CurrencyConverter
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model


def get_crypto_data():
    """ Get crypto currency data from CoinMarketCap 
    
    Returns:
        pandas.DataFrame -- DataFrame containing crypto currency data
    """
    if exists(f"{CRYPTO_CURRENCY}_all_time.csv"):
        print(f"[*]\tLoading {CRYPTO_CURRENCY} data from file")
        return pd.read_csv(f"{CRYPTO_CURRENCY}_all_time.csv")
    print("[*]\t Getting crypto data")
    # initialise scraper without time interval
    scraper = CmcScraper(CRYPTO_CURRENCY)

    # get raw data as list of list
    headers, data = scraper.get_data()

    # get data in a json format
    xrp_json_data = scraper.get_data("json")

    # export the data as csv file, you can also pass optional `name` parameter
    scraper.export("csv", name=f"{CRYPTO_CURRENCY}_all_time")

    # Pandas dataFrame for the same data
    df = scraper.get_dataframe()

    # High, Low, Open, Close, Volume, Adj Close
    data_frame = pd.DataFrame(data=df, columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume'])
    data_frame['Close'] = data_frame['Close'].apply(convert_to_gbp)
    return data_frame


def convert_to_gbp(value):
    """ Convert crypto currency value to GBP 
    
    Arguments:
        value {float} -- Crypto currency value
    
    Returns:
        float -- Converted value
    """
    # Initialise currency converter
    c = CurrencyConverter()
    # Convert to GBP
    value = c.convert(float(value), 'USD', 'GBP')
    # Return GBP value
    return value


def plot_actual_vs_prediction(actual, pred):
    """ Plot the predictions
    
    Arguments:
        actual {np.array} -- Actual values
        pred {np.array} -- Predicted values
    """
    print(f"[*]\tActual:\n{actual[-5:]}")
    print(f"[*]\tPredicted:\n{pred[-5:]}")
    plt.plot(actual[:], color = 'black', label='Actual Prices')
    plt.plot(pred[:], color='green', label = 'Predicted Prices')
    plt.title(f"{CRYPTO_CURRENCY} price prediction")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc = 'upper left')
    plt.savefig('actual_vs_prediction_all_time.png', bbox_inches='tight')


def plot_prediction(pred):
    """ Plot the prediction
    
    Arguments:
        pred {np.array} -- Predicted values
    """
    print("[*]\tPlotting prediction")
    plt.figure()
    plt.plot(pred, color = 'orange', label = 'Predicted Prices')
    plt.title(f"{CRYPTO_CURRENCY} price prediction")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc = 'upper left')
    plt.savefig(f'prediction_next_{PREDICTION_DAYS}_days.png', bbox_inches='tight')


def prepare_training_data(scaled_data):
    """ Prepare training data for the model
    
    Arguments:
        scaled_data {np.array} -- Scaled data
    
    Returns:
        x_train {np.array} -- Training data
        y_train {np.array} -- Training labels
    """
    print("[*]\tPreparing training data")
    # Prepare training data - Supervised learning
    x_train, y_train = [], []
    
    for x in range(PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[x-PREDICTION_DAYS:x, 0])
        y_train.append(scaled_data[x, 0])

    # Convert to np arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape training
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print("Training data prepared")
    return x_train, y_train


def create_model(x_train):
    """ Create a model with LSTM layers and dropout 
    
    Arguments:
        x_train {np.array} -- Training data
        y_train {np.array} -- Training labels
    
    Returns:
        model {keras.model} -- Model
    """
    print("[*]\tCreating predictive model")
    # Create Neural Network
    model = Sequential()

    # LSTM layers - recurrent layers to memorise stuff from each day - specialised on this sort of data - units = nodes
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2)) # Prevent overfitting
    model.add(LSTM(units=50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model


def train_model(x_train, y_train, model):
    """ Train the model
        
    Arguments:
        x_train {np.array} -- Training data
        y_train {np.array} -- Training labels
        model {keras.model} -- Model

    Returns:
        model {keras.model} -- Trained model
    """
    print("[*]\tTraining model")
    # Compile model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # Train model
    model.fit(x_train, y_train, batch_size = 32, epochs = 10)
    # Save model
    model.save(f'{CRYPTO_CURRENCY}_model.h5')
    print("[*]\tModel trained")
    return model


def evaluate_model(data, scaler, model):
    """ Evaluate the model by plotting actual and predicted prices
    
    Arguments:
        PREDICTION_DAYS {int} -- Number of days to predict
        data {pandas.DataFrame} -- DataFrame containing the data
        scaler {sklearn.preprocessing.MinMaxScaler} -- Scaler
        model {keras.model} -- Model
    """
    print("[*]\tEvaluating model")
    # Testing model
    test_data = get_crypto_data()

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    # Predictions
    x_test = [model_inputs[x - PREDICTION_DAYS : x, 0] for x in range(PREDICTION_DAYS, len(model_inputs))]


    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predicts prices
    prediction_prices = model.predict(x_test)

    # To get actual prices values from 0-1 values
    prediction_prices = scaler.inverse_transform(prediction_prices)
    plot_actual_vs_prediction(actual_prices, prediction_prices)


def predict(data, model, scaler):
    """ Predict the price for the next PREDICTION_DAYS
    
    Arguments:
        data {pandas.DataFrame} -- DataFrame containing the data
        model {keras.model} -- Model
        scaler {sklearn.preprocessing.MinMaxScaler} -- Scaler
        
    Returns:
        float -- Predicted price
    """
    print(f"[*]\tPredicting tomorrow's price for {CRYPTO_CURRENCY}")
    test_data = get_crypto_data()

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)
    # Predicting next day
    real_data = [model_inputs[len(model_inputs) + 1 - PREDICTION_DAYS: len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    preds = model.predict(real_data)
    print(len(preds))
    print(preds)
    input()
    # To get actual prices values from 0-1 values
    preds = scaler.inverse_transform(preds)

    plot_prediction(preds)

    final = format(float(preds[0]), '.8f') if preds[0] < 0.001 else float(preds[0])
    print(f"Tomorrow's price for {CRYPTO_CURRENCY} is £{final}")


def cls():
    """ Clear the screen """
    os.system('cls' if os.name=='nt' else 'clear')


def main():
    load_dotenv()
    global CRYPTO_CURRENCY
    global CURRENCY
    global PREDICTION_DAYS

    CRYPTO_CURRENCY = os.getenv('CRYPTO_CURRENCY')
    CURRENCY = os.getenv('CURRENCY')
    PREDICTION_DAYS = int(os.getenv('PREDICTION_DAYS'))

    cls()
    print("Welcome to crypto-analyst")
    data = get_crypto_data()
    
    # Prepare data
    # Scale data between 0 and 1
    scaler = MinMaxScaler(feature_range= (0, 1))
    # Normalise between -1 and 1
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    # Gets training data
    x_train, y_train = prepare_training_data(scaled_data)
    # If a trained model already exists
    if os.path.exists(f'{CRYPTO_CURRENCY}_model.h5'):
        print("[*]\tLoading model")
        model = load_model(f'{CRYPTO_CURRENCY}_model.h5')
    else:
        # Create model
        model = create_model(x_train)
        # Train model
        model = train_model(x_train, y_train, model)
    # Evaluate model
    evaluate_model(data, scaler, model)
    # Predict future prices
    predict(data, model, scaler)
    


if __name__ == '__main__':
    main()
