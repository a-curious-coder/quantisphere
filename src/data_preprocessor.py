""" This module preprocesses the data """
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    """ Preprocesses the data """
    @staticmethod
    def crypto(data):
        """ Preprocesses the crypto data """
        # Remove Market Cap column
        data.drop(columns=['Market Cap'], inplace=True)
        # Convert Date column to string
        data['Date'] = data['Date'].astype(str)
        # Reformat date from yyyy-mm-dd to dd-mm-yyyy
        data['Date'] = data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%d/%m/%Y'))
        
        # Round values to 2 decimal places and convert to float if the values are more than 10 (i.e. 10.00)
        data['Open'] = data['Open'].apply(lambda x: round(float(x), 2) if float(x) > 10 else float(x))
        data['High'] = data['High'].apply(lambda x: round(float(x), 2) if float(x) > 10 else float(x))
        data['Low'] = data['Low'].apply(lambda x: round(float(x), 2) if float(x) > 10 else float(x))
        data['Close'] = data['Close'].apply(lambda x: round(float(x), 2) if float(x) > 10 else float(x))

    @staticmethod
    def stocks(data):
        """ Preprocesses the stock data """
        # Reformat date from yyyy-mm-dd to dd-mm-yyyy
        data.index = data.index.strftime('%d/%m/%Y')
        # Change number of decimals to 2
        data['Open'] = data['Open'].apply(lambda x: round(float(x), 2) if float(x) > 10 else float(x))
        data['High'] = data['High'].apply(lambda x: round(float(x), 2) if float(x) > 10 else float(x))
        data['Low'] = data['Low'].apply(lambda x: round(float(x), 2) if float(x) > 10 else float(x))
        data['Close'] = data['Close'].apply(lambda x: round(float(x), 2) if float(x) > 10 else float(x))
        # Drop Dividends and Stock Splits columns
        data.drop(columns=['Dividends', 'Stock Splits', 'Volume'], inplace=True)
        return data

    def scale_data(self, data):
        """ Scales the data """
        scaler = MinMaxScaler()
        data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        return scaler, data