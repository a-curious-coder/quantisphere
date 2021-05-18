import numpy as np
import pandas as pd
import pandas as pandas_datareader
import matplotlib.pyplot as plt
# import pandas_datareader as web
from cryptocmd import CmcScraper
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

crypto_currency = "ELONGATE"
ac = 'GBP'

def get_data(crypto_currency):
    # initialise scraper without time interval
    scraper = CmcScraper(crypto_currency)

    # get raw data as list of list
    headers, data = scraper.get_data()

    # get data in a json format
    xrp_json_data = scraper.get_data("json")

    # export the data as csv file, you can also pass optional `name` parameter
    scraper.export("csv", name=f"{crypto_currency}_all_time")

    # Pandas dataFrame for the same data
    df = scraper.get_dataframe()

    
    # print(type(df), df.shape)
    # High, Low, Open, Close, Volume, Adj Close
    data_frame = pd.DataFrame(data=df, columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume'])
    data_frame['Close'] = data_frame['Close'].apply(convert_to_gbp)
    input(data_frame.head())
    print(f"Historical data for {crypto_currency} successfully retrieved.")
    return data_frame
    
def convert_to_gbp(value):
    from currency_converter import CurrencyConverter
    c = CurrencyConverter()
    value = c.convert(float(value), 'USD', 'GBP')
    return value

def prepare_training_data(prediction_days, scaled_data):
    # Prepare training data - Supervised learning
    x_train, y_train = [], []
    
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    # Convert to np arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape training
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print("Training data prepared")
    return x_train, y_train

def get_model(x_train, y_train):
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

    model.compile(optimizer = 'adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size = 32)
    print("Sequential model generated.")
    return model

def test_model(prediction_days, data, scaler, model):
    print("Testing predictive model")
    # Testing model
    test_data = get_data(crypto_currency)

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    # Predictions
    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x -  prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predicts prices
    prediction_prices = model.predict(x_test)

    # To get actual prices values from 0-1 values
    prediction_prices = scaler.inverse_transform(prediction_prices)
    print(f"Actual:\n{actual_prices}")
    print(f"Predicted:\n{prediction_prices}")
    print(f"Prediction prices\n{prediction_prices}")
    plt.plot(actual_prices, color = 'black', label='Actual Prices')
    plt.plot(prediction_prices, color='green', label = 'Predicted Prices')
    plt.title(f"{crypto_currency} price prediction")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc = 'upper left')
    plt.show()

def use_model(data, model, prediction_days, scaler):
    print(f"Predicting tomorrow's price for {crypto_currency}")
    test_data = get_data(crypto_currency)
    # print(test_data)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)
    # Predicting next day
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days: len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    # To get actual prices values from 0-1 values
    prediction = scaler.inverse_transform(prediction)

    # print(type(prediction[0]))
    if crypto_currency == 'ELONGATE':
        final = format(float(prediction[0]), '.8f')
    else:
        final = float(prediction[0])
    print(f"£{final}")

def main():
    print("Welcome to crypto-analyst")
    data = get_data(crypto_currency)
    # Prepare data
    # Squishes data to values between 0-1, more accurate.
    scaler = MinMaxScaler(feature_range= (0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    prediction_days = 30
    # Gets training data
    x_train, y_train = prepare_training_data(prediction_days, scaled_data)
    model = get_model(x_train, y_train)
    # test_model(prediction_days, data, scaler, model)
    use_model(data, model, prediction_days, scaler)

if __name__ == '__main__':
    main()