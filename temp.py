from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import time

crypto_currency = 'ELONGATE'
against_currency = 'GBP'
currency_symbol = '$'


def current_price(url, parameters, headers, session):
    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
        input(data)
        for i in range(0, 5000):
            if data['data'][i]['symbol'] == crypto_currency:
                price = data['data'][i]['quote']['GBP']['price']
                price = "%.16f" % float(price)
                return price
    
        # print(f"{data['data'][0]['symbol']}")
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(f"fuck: {e}\n")

def main():
    if against_currency == 'GBP':
        currency_symbol = '£'
    elif against_currency == 'USD':
        currency_symbol = '$'

    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
        'start':'1',
        'limit':'5000',
        'convert': 'GBP'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '8b969992-3bc0-47a4-9f61-d75fe944a7d3',
    }

    session = Session()
    session.headers.update(headers)
    print("Welcome to ELONGATE PRICE TRACKER")
    oldprice = 0
    start = time.time()
    while True:
        time.sleep(1)
        price = current_price(url, parameters, headers, session)
        if oldprice != price:
            end = time.time()
            duration = end-start;
            start = time.time()
            oldprice = price
            print(f"£{price}\t{duration}s")

if __name__ == "__main__":
    main()