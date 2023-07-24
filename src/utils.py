
import os

def clear_and_show_title():
    """ Clears the screen and shows the title """
    os.system('cls' if os.name == 'nt' else 'clear')
    print('===================================')
    print('||          WELCOME TO            ||')
    print('||        CRYPTO-ANALYST          ||')
    print('||         VERSION 1.0.0          ||')
    print('===================================')

# Create a function that ensures specific folders exist
def ensure_folders_exist():
    """ Ensures that the data, models, predictions, results and logs folders exist """
    folders = ['images', 'data', 'models', 'predictions', 'results', 'logs']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)