""" Data collector crypto currency data from CoinMarketCap """
import os
import numpy as np
import pandas as pd
from cryptocmd import CmcScraper
from currency_converter import CurrencyConverter
from sklearn.preprocessing import MinMaxScaler


class DataCollector:
    """ Data collector crypto currency data from CoinMarketCap """
    def __init__(self, crypto_currency: str):
        """ Initializes the DataCollector class
        Parameters
        ----------
        crypto_currency : str
            Name of the crypto currency
        """
        self.crypto_currency = crypto_currency
        self.all_time_data_file = f"{crypto_currency}_all_time.csv"
        self.crypto_data = pd.DataFrame()

    # Create a function that creates a training/test data set from the crypto data
    def create_data(self, data: pd.DataFrame) -> (np.array, np.array):
        """ Create training and testing data for the model
        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe with crypto currency data
        Returns
        -------
        training_data : np.array
            Training data for the model
        testing_data : np.array
            Testing data for the model
        """
        print("[INFO]\tPreparing training data")
        # Scale data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Normalise between -1 and 1
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        # Split data into training and testing
        training_data = scaled_data[:int(len(scaled_data) * 0.8)]
        testing_data = scaled_data[int(len(scaled_data) * 0.8) - 60:]
        return training_data, testing_data

    def get_training_data(self):
        """ Prepare training data for the model x_train and y_train
        Returns
        -------
        x_train : np.array
            Training data for the model
        y_train : np.array
            Training data for the model
        """
        # Gets training data
        x_train, y_train = self.create_data(self.crypto_data)
        x_train = [x_train[x - 60: x, 0] for x in range(60, len(x_train))]
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        return x_train, y_train

    def get_crypto_data(self) -> pd.DataFrame:
        """ Get crypto currency data from CoinMarketCap
        Returns
        -------
        crypto_data : pandas.DataFrame
            Dataframe with crypto currency data
        """
        # If the file already exists
        if os.path.exists(self.all_time_data_file):
            print(f"[INFO]\\tLoading {self.crypto_currency} data from file")
            self.crypto_data = pd.read_csv(self.all_time_data_file)
        else:
            print("[INFO]\\t Getting crypto data")
            # initialise scraper without time interval
            scraper = CmcScraper(self.crypto_currency)

            # Pandas dataFrame for the same data
            self.crypto_data = scraper.get_dataframe()

            # Set headers for data
            self.crypto_data = pd.DataFrame(
                data=self.crypto_data,
                columns=[
                    'Date',
                    'High',
                    'Low',
                    'Open',
                    'Close',
                    'Volume'
                ]
            )
            self.crypto_data.to_csv(self.all_time_data_file, index=False)
            self.crypto_data['Open'] = self.crypto_data['Open'].apply(self.convert_to_gbp)
            self.crypto_data['High'] = self.crypto_data['High'].apply(self.convert_to_gbp)
            self.crypto_data['Low'] = self.crypto_data['Low'].apply(self.convert_to_gbp)
            self.crypto_data['Close'] = self.crypto_data['Close'].apply(self.convert_to_gbp)
            # Save data to file
            self.crypto_data.to_csv(self.all_time_data_file, index=False)
        return self.crypto_data

    @staticmethod
    def convert_to_gbp(value: float) -> float:
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
        # NOTE: This function is here because the data collected from the API is in USD
        converter = CurrencyConverter()
        # Convert to GBP
        value = converter.convert(float(value), 'USD', 'GBP')
        return value
