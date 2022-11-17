# Crypto Analyst
Price data is scraped from CoinMarketCap, processed and used to predict the future price of a user-specified crypto currency over a set number of days. This project uses Deep Learning, incorporating LSTM layers in the neural network to best analyse and predict new prices according to the time series price data.

![Image](media/crypto_landscape.jpg)


# Setup

Create your virtual environment and install the libraries from requirements.txt:

## Windows
    $ python virtualenv .venv
    $ .venv\Scripts\activate
    $ pip install -r requirements.txt

## Linux / Mac

    $ python virtualenv .venv
    $ source .venv/bin/activate
    $ pip install -r requirements.txt

# Methodology
Once data was collected, I had to figure out the best way to split the time-series data into train/test sets. I decided to use the last 20% of the data as the test set, and the remaining 80% as the training set. This is because it gives the model the most recent data to train on, and the most recent data to test on. This is the most accurate way to test the model, as it is the most recent data that we are trying to predict.