import os
import sys

# import numpy as np
# import pandas as pd
import warnings

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
from keras.models import load_model
from crypto_analyst import CryptoAnalyst
from crypto_analyst import LSTMModel
from data_collector import DataCollector

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cls():
    """ Clear the screen """
    os.system('cls' if os.name=='nt' else 'clear')


def show_title():
    os.system('cls' if os.name == 'nt' else 'clear')
    print('===================================')
    print('||          WELCOME TO            ||')
    print('||        CRYPTO-ANALYST          ||')
    print('||         VERSION 1.0.0          ||')
    print('===================================')


def main():
    """ Main function """
    crypto = "BTC"
    prediction_days = 3
    start_date, end_date = "15-10-2021", "15-10-2022"
    data_collector = DataCollector(crypto, start_date, end_date)
    
    # We want to collect training and testing datasets for the user chosen crypto currency
    # TODO: Validate whether the crypto-currency exists.
    cls()
    show_title()

    data_collector.prepare_train_test_datasets(data_collector.load_crypto_data())
    # Collects crypto currency data from the API
    training_data, testing_data = data_collector.get_train_test_data()

    # Create the model
    model = LSTMModel(training_data, prediction_days)

    # Train the model
    model.train()

    # Save the model
    model.save("models/lstm_model.h5")

    # Load the model
    model = load_model("models/lstm_model.h5")

    # Create the crypto analyst
    crypto_analyst = CryptoAnalyst(model, testing_data, prediction_days)

    # Get the predictions
    predictions = crypto_analyst.get_predictions()

    # Get the actual values
    actual_values = crypto_analyst.get_actual()

    print("Predictions: ", predictions)
    print("Actual Values: ", actual_values)

    


    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
