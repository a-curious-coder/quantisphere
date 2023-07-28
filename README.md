# Quantisphere
Price data is scraped from CoinMarketCap, processed and used to predict the future price of a user-specified crypto currency over a set number of days. This project uses Deep Learning, incorporating LSTM layers in the neural network to best analyse and predict new prices according to the time series price data.

![Image](media/crypto_landscape.jpg)

# How does it work?
- Collect Data
    - Use yfinance or coinmarketcap to collect data.
    - Perform slight pre-processing on the data for model preparation and human readability.
- Save Data
    - Save the pre-processed data to a CSV file for future use.
- Split Data
    - Split the data into train and test sets.
- Train Model
    - Train the model using the training set.
- Save Model
    - Save the trained model to storage for future use.

## Approaches 
- [ ] LSTM
- [ ] ARIMA
- [ ] Prophet
- [ ] Multimodal (LSTM + ARIMA + Prophet)?
### What alternate approaches can we take?
With the release of modern Natural Language Processing models (ChatGPT), it's a very achievable goal to bring in analysis of up-to-date and real world news articles to predict the price of a stock or crypto currency. This is a very interesting approach, and I will be looking into it in the future.
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