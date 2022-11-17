""" Plotter Class """
import os
from matplotlib import pyplot as plt

class Plotter:
    """ Plotter Class """
    def __init__(self, crypto, prediction_days, show=False):
        self.crypto = crypto
        self.prediction_days = prediction_days
        self.show = show
        # Create plots folder in working directory
        if not os.path.exists("plots"):
            os.mkdir("plots")

    def actual_vs_prediction(self, actual, pred):
        """ Plot the predictions
        Parameters
        ----------
        crypto : str
            Crypto currency
        actual : np.array
            Actual values
        pred : np.array
            Predicted values
        """
        print(f"[INFO]\tActual:\n{actual[-5:]}")
        print(f"[INFO]\tPredicted:\n{pred[-5:]}")

        plt.plot(actual[:], color = 'black', label='Actual Prices')
        plt.plot(pred[:], color='green', label = 'Predicted Prices')
        plt.title(f"{self.crypto}'s price prediction")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc = 'upper left')
        plt.savefig('plots/actual_vs_prediction.png', bbox_inches='tight')

    def predictions(self, pred):
        """ Plot the prediction
        Parameters
        ----------
        crypto : str
            Crypto currency
        days : int
            Number of days to predict
        pred : np.array
            Predicted values
        """
        print("[INFO]\tPlotting prediction")
        plt.figure()
        plt.plot(pred, color = 'orange', label = 'Predicted Prices')
        plt.title(f"{self.crypto} price prediction")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend(loc = 'upper left')
        plt.savefig(f'plots/prediction_next_{self.prediction_days}_days.png', bbox_inches='tight')
