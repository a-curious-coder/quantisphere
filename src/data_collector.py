""" Data collector crypto currency data from CoinMarketCap """
import os
from datetime import datetime
import numpy as np
import pandas as pd

from cryptocmd import CmcScraper
from currency_converter import CurrencyConverter
from sklearn.preprocessing import MinMaxScaler


class DataCollector:
    """ Data collector crypto currency data from CoinMarketCap """
    def __init__(self, crypto_currency: str, start_date = None, end_date = None, debug = False):
        """ Initializes the DataCollector class
        Parameters
        ----------
        crypto_currency : str
            Name of the crypto currency
        """
        self.crypto_currency = crypto_currency
        self.start_date = start_date
        self.end_date = end_date
        if start_date is not None and end_date is not None:
            self.date_range = True
            self.file_name = f"{crypto_currency}_{start_date}_{end_date}.csv"
        else:
            self.date_range = False
            self.file_name = f"{crypto_currency}_all_time.csv"
        self.crypto_data = pd.DataFrame()
        self.training_data = None
        self.testing_data = None
        self.debug = debug
        self.load_crypto_data()
        self.prepare_train_test_datasets()

    def validate_crypto(self) -> str:
        # TODO: Validate crypto-currency user gives is valid
        return

    def prepare_train_test_datasets(self, train_pct: float = 0.8, feature_range: tuple = (0, 1), fillna: bool = False, fill_value: float = 0) -> (np.array, np.array):
        """ Prepares training and testing data for the model
        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe with crypto currency data
        train_pct : float, optional
            Percentage of data to be used for training, by default 0.8
        feature_range : tuple, optional
            Range for scaling data, by default (0, 1)
        fillna : bool, optional
            Whether or not to fill missing values, by default False
        fill_value : float, optional
            Value to fill missing values with, by default 0
        Returns
        -------
        training_data : np.array
            Training data for the model
        testing_data : np.array
            Testing data for the model
        """
        try:
            # Scale data
            scaler = MinMaxScaler(feature_range=feature_range)
            scaled_data = scaler.fit_transform(self.crypto_data['Close'].values.reshape(-1, 1))
            # Split data into training and testing
            train_size = int(len(scaled_data) * train_pct)
            self.training_data = scaled_data[:train_size]
            self.testing_data = scaled_data[train_size:]
        except Exception as e:
            self._print(f"[ERROR]\t{e}")

    def get_train_test_data(self):
        return self.training_data, self.testing_data

    def load_crypto_data(self) -> pd.DataFrame:
        """ Get crypto currency data from CoinMarketCap
        Returns
        -------
        crypto_data : pandas.DataFrame
            Dataframe with crypto currency data
        """
        if os.path.exists(self.file_name):
            self._print(f"[INFO]\tLoading \'{self.crypto_currency}\' data spanning over {self._duration_between_dates()}")
            self.crypto_data = pd.read_csv(self.file_name)
        else:
            if self.date_range:
                self._print(f"[INFO]\tCollecting \'{self.crypto_currency}\' data between {self.start_date} and {self.end_date}; Spanning over {self._duration_between_dates()}")
            else:
                self._print(f"[INFO]\tCollecting all cryptocurrency data for \'{self.crypto_currency}\'")
            self._get_crypto_data()


    def _get_crypto_data(self):
        """Fetches cryptocurrency data from CoinMarketCap and saves it to a file."""
        try:
            # Initialise scraper which will automatically collect the given cryptocurrency's data
            scraper = CmcScraper(self.crypto_currency, start_date=self.start_date, end_date=self.end_date)

            # Get the data from the scraper object as a Pandas Dataframe
            self.crypto_data = scraper.get_dataframe()

            # Filter relevant columns and convert data to GBP
            self.crypto_data = self._filter_and_convert(self.crypto_data)

            # Save data to file
            self.crypto_data.to_csv(self.file_name, index=False)
        except Exception as e:
            self._print(f"[ERROR]\t{e}")
            self.crypto_data = None

    def _filter_and_convert(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Filter and convert relevant columns in the crypto currency data
        Parameters
        ----------
        data : pandas.DataFrame
            The dataframe with crypto currency data
        Returns
        -------
        filtered_data : pandas.DataFrame
            The filtered dataframe with relevant columns converted
        """
        # Set headers for data
        filtered_data = pd.DataFrame(
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
        filtered_data['Open'] = filtered_data['Open'].apply(self.convert_to_gbp)
        filtered_data['High'] = filtered_data['High'].apply(self.convert_to_gbp)
        filtered_data['Low'] = filtered_data['Low'].apply(self.convert_to_gbp)
        filtered_data['Close'] = filtered_data['Close'].apply(self.convert_to_gbp)
        filtered_data.to_csv(self.file_name, index=False)
        return filtered_data

    def _duration_between_dates(self):
        # TODO: Validate start date is before end date
        d1 = datetime.strptime(self.start_date, "%d-%m-%Y")
        d2 = datetime.strptime(self.end_date, "%d-%m-%Y")

        duration = d2 - d1

        years = duration.days // 365
        months = (duration.days % 365) // 30
        days = (duration.days % 365) % 30

        duration_str = ""

        if years == 1:
            duration_str += f"{years} year"
        elif years > 1:
            duration_str += f"{years} years"

        if months == 1:
            if duration_str:
                duration_str += ", "
            duration_str += f"{months} month"
        elif months > 1:
            if duration_str:
                duration_str += ", "
            duration_str += f"{months} months"

        if days == 1:
            if duration_str:
                duration_str += " and "
            duration_str += f"{days} day"
        elif days > 1:
            if duration_str:
                duration_str += " and "
            duration_str += f"{days} days"

        return duration_str
        
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

    # Create overload print function using debug flag
    def _print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

def show_title():
    os.system('cls' if os.name == 'nt' else 'clear')
    print('===================================')
    print('||          WELCOME TO            ||')
    print('||        CRYPTO-ANALYST          ||')
    print('||       [DATA COLLECTOR]         ||')
    print('||         VERSION 1.0.0          ||')
    print('===================================')


def main():
    show_title()
    start_date, end_date = "15-10-2021", "15-10-2022"
    # Initialize data collector
    data_collector = DataCollector("BTC", start_date, end_date, debug=True)

    # Fetch training and test sets
    training_data, testing_data = data_collector.get_train_test_data()

    # Print results
    print(f"[INFO]\t[{training_data.shape[0]} | {testing_data.shape[0]}]")


if __name__ == "__main__":
    main()