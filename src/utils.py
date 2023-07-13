
import os

def clear_and_show_title():
    """ Clears the screen and shows the title """
    os.system('cls' if os.name == 'nt' else 'clear')
    print('===================================')
    print('||          WELCOME TO            ||')
    print('||        CRYPTO-ANALYST          ||')
    print('||         VERSION 1.0.0          ||')
    print('===================================')

    def convert_to_gbp(value: float) -> float:
        """ Convert crypto currency value to GBP
        Parameters
        ----------
        value : float
            Crypto currency value
        Returns
        -------
        value : float
            Crypto currency value in GBP
        """
        # NOTE: This function is here because the data collected from the API is in USD
        converter = CurrencyConverter()
        # Convert to GBP
        value = converter.convert(float(value), 'USD', 'GBP')
        return value