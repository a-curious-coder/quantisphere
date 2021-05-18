from selenium import webdriver
import bs4
import pandas as pd
import requests
from currency_converter import CurrencyConverter
import time

currency = 'GBP'




def current_price(url):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    results = soup.find(id='__next')
    # print(results.prettify())
    price = results.find('div', class_='priceValue___11gHJ')
    price = price.text
    usd_price = price
    if "$" in price:
        price = price.replace("$", "")
    if "$" in usd_price:
        usd_price = float(usd_price.replace("$", ""))

    c = CurrencyConverter() 
    return ["%.10f" % float(c.convert(price, 'USD', currency)), "%.10f" % float(usd_price)]

def main():
    url = 'https://coinmarketcap.com/currencies/elongate/'
    print("Welcome to ELONGATE PRICE TRACKER")
    start = time.time()
    oldprice = 0
    while True:
        price = current_price(url)
        print(price[0])
        if oldprice != price[0]:
            end = time.time()
            duration = "%.2f" % float(end-start)
            start = time.time()
            oldprice = price[0]
            print(f"£{price[0]}\t${price[1]}\t{duration}s")

if __name__ == '__main__':
    main()