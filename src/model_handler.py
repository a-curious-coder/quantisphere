import os
from dataclasses import dataclass

import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential, load_model
from sklearn.metrics import mean_squared_error


@dataclass
class IncorrectModelTypeError(Exception):
    """Raised when the data type entered is incorrect"""

    model_type: str

    def __str__(self) -> str:
        return f"{self.model_type} is an incorrect model type; A model is either a 'LSTM' or 'GRU'."


class ModelHandler:
    model_name = "model.h5"
    valid_model_types = ["LSTM", "GRU"]
    filepath = f"models\{model_name}"

    def __init__(self, model_type: str = "LSTM"):
        if model_type not in self.valid_model_types:
            raise IncorrectModelTypeError(model_type)
        self.model_type = model_type

    def create_model(self, x_train, y_train, x_val, y_val):
        """Creates and trains the model"""

        self.model = Sequential()
        self.model.add(
            LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1))
        )
        self.model.add(LSTM(units=50))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        history = self.model.fit(
            x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32
        )
        return self.model, history

    def test_model(self, model, x_test):
        # Makes prediction and restores the original scale
        predicted_prices = model.predict(x_test)
        return predicted_prices

    def evaluate_model(self, test_data, predicted_prices):
        mse = mean_squared_error(test_data["Close"], predicted_prices)
        print("Mean Squared Error:", mse)

    def load_model(self) -> object:
        """Loads the model"""

        if os.path.exists(self.filepath):
            self.model = load_model(self.filepath)
        else:
            print(f"[INFO]\tNo model found at models/{self.model_name}")
            return None
        return self.model

    def save_model(self):
        """Saves the model"""
        print(f"[INFO]\tSaving model to {self.model_name}")
        self.model.save(f"models/{self.model_name}")
