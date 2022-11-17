""" Crypto Analyst"""
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from plotter import Plotter


class CryptoAnalyst():
    """ Crypto Analyst Class """
    def __init__(self, crypto, prediction_days, x_train, y_train, x_test, y_test):
        self.crypto = crypto
        self.prediction_days = prediction_days
        self.train_data = x_train
        self.train_labels = y_train
        self.test_data = x_test
        self.test_labels = y_test
        self.plotter = Plotter(self.crypto, self.prediction_days)

    def create_model(self):
        """ Create a model with LSTM layers and dropout 
    Returns
    -------
    model : tensorflow.keras.models.Sequential
        Model with LSTM layers and dropout
    """
        print("[INFO]\tCreating predictive model")
        # Create Neural Network
        model = Sequential()

        # LSTM layers - recurrent layers to memorise stuff from each day - specialised on this sort of data - units = nodes
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (self.train_data.shape[1], 1)))
        model.add(Dropout(0.2)) # Prevent overfitting
        model.add(LSTM(units=50, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        return model

    def train_model(self, model):
        """ Train the model
        Parameters
        ----------
        x_train : np.array
            Training data for model
        y_train : np.array
            Training labels for model
        model : tensorflow.keras.models.Sequential
            Model with LSTM layers and dropout
        Returns
        -------
        model : tensorflow.keras.models.Sequential
            Trained model
        """
        print("[INFO]\tTraining model")
        # Compile model
        model.compile(optimizer = 'adam', loss = 'mse')
        # Train model
        model.fit(self.train_data, self.train_labels, batch_size = 32, epochs = 10)
        # Save model
        model.save(f'{self.crypto}_model.h5')
        print("[INFO]\tModel trained")
        return model

    def run_model(self, data, model, scaler):
        """ Predict the price for the next prediction_days
        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe with crypto currency data
        model : tensorflow.keras.models.Sequential
            Trained model
        scaler : sklearn.preprocessing.MinMaxScaler
            Scaler for the data
        """
        print(f"[INFO]\tPredicting tomorrow's price for {self.crypto}")

        total_dataset = pd.concat((data['Close'], self.test_data['Close']), axis = 0)

        model_inputs = total_dataset[len(total_dataset) - len(self.test_data) - self.prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.fit_transform(model_inputs)
        # Predicting next day
        real_data = [model_inputs[len(model_inputs) + 1 - self.prediction_days: len(model_inputs) + 1, 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        preds = model.predict(real_data)
        print(len(preds))
        print(preds)
        input()
        # To get actual prices values from 0-1 values
        preds = scaler.inverse_transform(preds)

        self.plotter.predictions(preds)

        final = format(float(preds[0]), '.8f') if preds[0] < 0.001 else float(preds[0])
        print(f"Tomorrow's price for {self.crypto} is £{final}")

    def evaluate_model(self, model, scaler):
        """ Evaluate the model by plotting actual and predicted prices
        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe with crypto currency data
        scaler : sklearn.preprocessing.MinMaxScaler
            Scaler for the data
        model : tensorflow.keras.models.Sequential
            Trained model
        """
        print("[INFO]\tEvaluating model")

        # Predicts prices
        prediction_prices = model.predict(self.test_data)
        # To get actual prices values from 0-1 values
        prediction_prices = scaler.inverse_transform(prediction_prices)
        self.plotter.actual_vs_prediction(self.test_labels, prediction_prices)
