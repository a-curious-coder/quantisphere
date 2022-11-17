import os
import sys

import numpy as np
import pandas as pd
from cryptocmd import CmcScraper
from currency_converter import CurrencyConverter
from sklearn.preprocessing import MinMaxScaler

from crypto_analyst import CryptoAnalyst


def cls():
    """ Clear the screen """
    os.system('cls' if os.name=='nt' else 'clear')


def convert_to_gbp(value):
    """ Convert crypto currency value to GBP
    Parameters
    ----------
    value : float
        Crypto currency value
    Returns
    -------
    value : float
        Crypto currency value in GBP
    """
    converter = CurrencyConverter()
    # Convert to GBP
    value = converter.convert(float(value), 'USD', 'GBP')
    return value


def get_crypto_data(crypto_currency):
    """ Get crypto currency data from CoinMarketCap
    Returns
    -------
    crypto_data : pandas.DataFrame
        Dataframe with crypto currency data
    """
    # If the file already exists
    if os.path.exists(f"{crypto_currency}_all_time.csv"):
        print(f"[INFO]\tLoading {crypto_currency} data from file")
        return pd.read_csv(f"{crypto_currency}_all_time.csv")

    print("[INFO]\t Getting crypto data")
    # initialise scraper without time interval
    scraper = CmcScraper(crypto_currency)

    # export the data as csv file, you can also pass optional `name` parameter
    # scraper.export("csv", name=f"{crypto_currency}_all_time")

    # Pandas dataFrame for the same data
    crypto_data = scraper.get_dataframe()

    # Set headers for data
    crypto_data = pd.DataFrame(
        data=crypto_data,
        columns = [
            'Date',
            'High',
            'Low',
            'Open',
            'Close',
            'Volume'
        ]
    )
    crypto_data.to_csv(f"{crypto_currency}_all_time.csv", index=False)
    crypto_data['Open'] = crypto_data['Open'].apply(convert_to_gbp)
    crypto_data['High'] = crypto_data['High'].apply(convert_to_gbp)
    crypto_data['Low'] = crypto_data['Low'].apply(convert_to_gbp)
    crypto_data['Close'] = crypto_data['Close'].apply(convert_to_gbp)
    # Save data to file
    crypto_data.to_csv(f"{crypto_currency}_all_time.csv", index=False)
    return crypto_data


def get_training_data(data):
    """ Prepare training data for the model
    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with crypto currency data
    Returns
    -------
    training_data : np.array
        Training data for the model
    """
    print("[INFO]\tPreparing training data")
    # Convert dataframe into dictionary of columns
    data = data.to_dict('list')
    # Get the close price
    close_data = data['Close']
    print(len(close_data))
    # Create training data
    training_data = []
    for i in range(60, len(close_data)):
        training_data.append(close_data[i-60:i])
    # Convert to numpy array
    training_data = np.array(training_data)
    return training_data

    # Prepare training data - Supervised learning
    x_train, y_train = [], []
    
    for day in range(3, len(data)):
        x_train.append(data[day-3:day, 0])
        y_train.append(data[day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape training
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print("[INFO] Training data prepared")
    return x_train, y_train


def main():
    """ Main function """
    crypto = "ada"

    cls()
    print("Welcome to crypto-analyst")
    if not os.path.exists('data.csv'):
        # Collect data and split into training and testing
        data = get_crypto_data(crypto)
    else:
        data = pd.read_csv('data.csv')

    # Scale data between 0 and 1
    scaler = MinMaxScaler(feature_range= (0, 1))
    # Normalise between -1 and 1
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    # Split data into training and testing
    training_data = scaled_data[:int(len(scaled_data) * 0.8)]
    testing_data = scaled_data[int(len(scaled_data) * 0.8) - 60:]
    
    # Gets training data
    x_train, y_train = get_training_data(scaled_data)

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    # Predictions
    x_test = [model_inputs[x - prediction_days : x, 0] for x in range(prediction_days, len(model_inputs))]

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    crypto_analyst = CryptoAnalyst(crypto, 3, x_train, y_train, x_test, y_test)
    # If a trained model already exists
    if os.path.exists(f'{crypto}_model.h5'):
        print("[INFO]\tLoading model")
        model = load_model(f'{crypto}_model.h5')
    else:
        # Create model
        model = create_model(x_train)
        # Train model
        model = train_model(x_train, y_train, model)
    # Evaluate model
    evaluate_model(data, scaler, model)
    # Predict future prices
    test_model(data, model, scaler)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
