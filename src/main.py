import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from crypto_analyst import CryptoAnalyst
from crypto_analyst import LSTMModel
from data_collector import DataCollector

def cls():
    """ Clear the screen """
    os.system('cls' if os.name=='nt' else 'clear')


def main():
    """ Main function """
    crypto = "ADA"
    prediction_days = 3
    data_collector = DataCollector(crypto)

    cls()
    print("Welcome to crypto-analyst")
    data = data_collector.get_crypto_data()

    # Scale data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Normalise between -1 and 1
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    # Split data into training and testing
    training_data = scaled_data[:int(len(scaled_data) * 0.8)]
    testing_data = scaled_data[int(len(scaled_data) * 0.8) - 60:]

    # Gets training data
    x_train, y_train = data_collector.get_training_data()

    total_dataset = pd.concat((data['Close'], testing_data), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(testing_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    # Predictions
    x_test = [model_inputs[x - prediction_days: x, 0] for x in range(prediction_days, len(model_inputs))]

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    crypto_analyst = CryptoAnalyst(crypto, prediction_days, x_train, y_train, x_test, testing_data)
    mod = LSTMModel()
    # If a trained model already exists
    if os.path.exists(f'{crypto}_model.h5'):
        print("[INFO]\\tLoading model")
        model = load_model(f'{crypto}_model.h5')
    else:
        # Create model
        model = mod.create_model(data_shape = x_train.shape[1])
        # Train model
        model = mod.train_model(model = model, x_train = x_train, y_train = y_train, crypto=crypto)
    # Evaluate model
    crypto_analyst.evaluate_model(model, scaler)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
