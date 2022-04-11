# Crypto Analayst
This project scrapes price data from CoinMarketCap to predict the future price over a user-specified number of days. This project uses a Long Short Term Memory Neural Network that works with time series price data.
# Setup

First, we need to set up a virtual environment. If you do not have 'virtualenv' installed on your system, follow the steps below corresponding to your operating system.

```bash
# Windows and Mac
pip install virtualenv
```

With virtualenv installed, we need to create the virtual environment.

```bash
# Windows and Mac
virtualenv <virtual environment name>
```

The virtual environment is created. We need to initialise the virtual environment in order to install all of the preliminary requirements to run this project.

```bash
# Windows
<virtual environment name>\Scripts\activate.bat

# Mac
source <virtual environment name>/bin/activate
```

With the virtual environment activated, we need to install the requirements from the 'requirements.txt' file

```bash
# Windows and Mac
pip install -r requirements.txt
```

Once all of the requirements are installed, run 'main.py'

```bash
python main.py
```
