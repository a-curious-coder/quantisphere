""" Crypto Analyst"""
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from plotter import Plotter


# Extract Class: LSTM Model Creation and Training
class LSTMModel:
    """
    LSTM Model Creation and Training
    """
    def __init__(self, training_data: np.array, prediction_days: int = 3):
        """
        Initializes the LSTMModel class

        Parameters
        ----------
        training_data : np.array
            Training data for the model
        prediction_days : int, optional
            Number of days to predict, by default 3
        """
        self.training_data = training_data
        self.prediction_days = prediction_days
        self.scaler = None
        # If model already exists, load it
        try:
            # HACK: Hardcoded path
            self.model = load_model(f"models/lstm_model.h5")
        # If model doesn't exist, create it
        except OSError:
            self.model = self.create_model(len(training_data))

    @staticmethod
    def create_model(data_shape: int, units: int = 50, dropout: float = 0.2) -> Sequential:
        """
        Creates a neural network with LSTM layers and dropout

        Parameters
        ----------
        data_shape : int
            Shape of input data
        units : int, optional
            Number of units in LSTM layer, by default 50
        dropout : float, optional
            Dropout rate, by default 0.2

        Returns
        -------
        Sequential
            Model with LSTM layers and dropout
        """
        model = Sequential()
        # LSTM layers - recurrent layers to memorise stuff from each day - specialised on this sort of data - units = nodes
        model.add(LSTM(units=units, return_sequences=True, input_shape=(data_shape, 1)))
        model.add(Dropout(dropout)) # Prevent overfitting
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units))
        model.add(Dropout(dropout))
        model.add(Dense(units=1))
        return model

    def train(self, epochs: int = 25, batch_size: int = 32) -> None:
        """
        Trains the model

        Parameters
        ----------
        epochs : int, optional
            Number of epochs to train for, by default 25
        batch_size : int, optional
            Batch size, by default 32
        """
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.training_data, self.training_data, epochs=epochs, batch_size=batch_size)

    def save(self, model_path: str) -> None:
        """
        Saves the model

        Parameters
        ----------
        model_name : str
            Name of the model
        """
        self.model.save(model_path)

    

class CryptoAnalyst:
    """
    Crypto Analyst Class
    """
    def __init__(self, model_name: str, crypto_data: pd.DataFrame, prediction_days: int = 3, train_pct: float = 0.8, feature_range: tuple = (0, 1), fillna: bool = False, fill_value: float = 0):
        """
        Initializes the CryptoAnalyst class

        Parameters
        ----------
        model_name : str
            Name of the model
        crypto_data : pandas.DataFrame
            Dataframe with crypto currency data
        prediction_days : int, optional
            Number of days to predict, by default 3
        train_pct : float, optional
            Percentage of data to be used for training, by default 0.8
        feature_range : tuple, optional
            Range for scaling data, by default (0, 1)
        fillna : bool, optional
            Whether or not to fill missing values, by default False
        fill_value : float, optional
            Value to fill missing values with, by default 0
        """
        self.model_name = model_name
        self.crypto_data = crypto_data
        self.prediction_days = prediction_days
        self.plotter = Plotter('BTC', prediction_days=prediction_days)
        # Plot the predicted and actual values
        self.plotter.actual_vs_prediction(self.get_actual(), self.get_predictions())
        self.plotter.predictions(self.get_predictions())

    def get_predictions(self, days: int = 30) -> pd.DataFrame:
        """
        Gets predictions for the next 30 days

        Parameters
        ----------
        days : int, optional
            Number of days to predict, by default 30

        Returns
        -------
        pandas.DataFrame
            Dataframe with predictions
        """
        predictions = self.model.predict(self.testing_data)
        predictions = self.model.scaler.inverse_transform(predictions)
        predictions = pd.DataFrame(predictions, columns=['Prediction'])
        predictions['Date'] = self.crypto_data['Date'].tail(days)
        predictions = predictions.set_index('Date')
        return predictions

    def get_actual(self, days: int = 30) -> pd.DataFrame:
        """
        Gets actual values for the next 30 days

        Parameters
        ----------
        days : int, optional
            Number of days to predict, by default 30

        Returns
        -------
        pandas.DataFrame
            Dataframe with actual values
        """
        actual = self.crypto_data['Close'].tail(days)
        actual = pd.DataFrame(actual, columns=['Actual'])
        actual['Date'] = self.crypto_data['Date'].tail(days)
        actual = actual.set_index('Date')
        return actual