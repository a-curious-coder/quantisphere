import numpy as np
import pandas as pd
import pandas as pandas_datareader
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

cc = 'BTC'
ac = 'GBP'

crypto_currency = "ELONGATE"
from cryptocmd import CmcScraper

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

    # print(df.head())
    # print(type(df), df.shape)
    # High, Low, Open, Close, Volume, Adj Close
    data_frame = pd.DataFrame(data=df, columns = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume'])
    return data_frame
    
data_frame = get_data(crypto_currency)
# print(data_frame.columns)
start = dt.datetime(2016,1,1)
end = dt.datetime.now()

data = web.DataReader(f"{cc}-{ac}", 'yahoo', start, end)

# print(data_frame.head())
data = data_frame.copy()

# Prepare data
# Squishes data to values between 0-1, more accurate.
scaler = MinMaxScaler(feature_range= (0, 1))
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 30

# Prepare training data
# Supervised learning
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

# Convert to np arrays
x_train, y_train = np.array(x_train), np.array(y_train)
# Reshape training
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

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


# Testing model
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

# test_data = web.DataReader(f"{cc}-{ac}", 'yahoo', test_start, test_end)
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

prediction_prices = model.predict(x_test)
# To get actual prices values from 0-1 values
prediction_prices = scaler.inverse_transform(prediction_prices)
print(f"Actual:\n{actual_prices}")
print(f"Predicted:\n{prediction_prices}")
print(f"Prediction prices\n{prediction_prices}")
plt.plot(actual_prices, color = 'black', label='Actual Prices')
plt.plot(prediction_prices, color='green', label = 'Predicted Prices')
plt.title(f"{cc} price prediction")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc = 'upper left')
plt.show()

# Predicting next day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days: len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
# predictions = []
# for i in range(0, 25):
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
#     predictions.append(prediction[0])
#     # print(prediction[0])
# mean_prediction = sum(predictions) / len(predictions)
# final = "%.10f" % float(mean_prediction)
# print(f"£{final}")

# plt.plot(prediction, color = 'orange', label = 'Real Data Predictions')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend(loc = 'upper left')
# plt.show()
print(type(prediction[0]))
final = format(float(prediction[0]), '.8f')
print(f"£{final}")
