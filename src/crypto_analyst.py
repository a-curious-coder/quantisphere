""" Crypto Analyst"""
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from plotter import Plotter


# Extract Class: LSTM Model Creation and Training
class LSTMModel:
    """
    LSTM Model Creation and Training
    """
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

    @staticmethod
    def train_model(model: Sequential, x_train: np.array, y_train: np.array, crypto: str) -> Sequential:
        """
        Trains the model

        Parameters
        ----------
        model : Sequential
            Model with LSTM layers and dropout
        x_train : np.array
            Training data for model
        y_train : np.array
            Training labels for model
        crypto : str
            Name of the cryptocurrency

        Returns
        -------
        Sequential
            Trained model
        """
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        # Train model
        model.fit(x_train, y_train, batch_size=32, epochs=10)
        # Save model
        model.save(f'{crypto}_model.h5')
        print("[INFO]\\tModel trained")
        return model


class CryptoAnalyst:
    """
    Crypto Analyst Class
    """
    def __init__(self, crypto: str, prediction_days: int, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
        """
        Initializes the CryptoAnalyst class

        Parameters
        ----------
        crypto : str
            Name of the cryptocurrency
        prediction_days : int
            Number of days to predict
        x_train : np.array
            Training data for model
        y_train : np.array
            Training labels for model
        x_test : np.array
            Testing data for model
        y_test : np.array
            Testing labels for model
        """
        self.crypto = crypto
        self.prediction_days = prediction_days
        self.train_data = x_train
        self.train_labels = y_train
        self.test_data = x_test
        self.test_labels = y_test
        self.plotter = Plotter(self.crypto, self.prediction_days)

    def run_model(self, data: pd.DataFrame, model: Sequential, scaler: MinMaxScaler) -> None:
        """
        Predicts the price for the next prediction_days

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with cryptocurrency data
        model : Sequential
            Trained model
        scaler : MinMaxScaler
            Scaler for the data
        """
        print(f"[INFO]\\tPredicting tomorrow's price for {self.crypto}")

        # Get the total dataset
        total_dataset = pd.concat((data['Close'], self.test_data['Close']), axis=0)

        # Get the inputs for the model
        model_inputs = total_dataset[len(total_dataset) - len(self.test_data) - self.prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.fit_transform(model_inputs)

        # Predicting next day
        real_data = [model_inputs[len(model_inputs) + 1 - self.prediction_days: len(model_inputs) + 1, 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        # Get predictions
        preds = model.predict(real_data)

        # To get actual price values from 0-1 values
        preds = scaler.inverse_transform(preds)

        # Plot predictions
        self.plotter.predictions(preds)

        # Format final prediction
        final = format(float(preds[0]), '.8f') if preds[0] < 0.001 else float(preds[0])
        print(f"Tomorrow's price for {self.crypto} is £{final}")

    def evaluate_model(self, model: Sequential, scaler: MinMaxScaler) -> None:
        """
        Evaluates the model by plotting actual and predicted prices

        Parameters
        ----------
        model : Sequential
            Trained model
        scaler : MinMaxScaler
            Scaler for the data
        """
        print("[INFO]\\tEvaluating model")

        # Predict prices
        prediction_prices = model.predict(self.test_data)

        # To get actual prices values from 0-1 values
        prediction_prices = scaler.inverse_transform(prediction_prices)

        # Plot actual vs predicted prices
        self.plotter.actual_vs_prediction(self.test_labels, prediction_prices)
