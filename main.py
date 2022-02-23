import json
import os

from violetBot import VioletBot


### FUNCTIONS ###

def display_title_bar():
    # Clears the terminal screen, and displays a title bar.

    print("**********************************************")
    print("***  Violet - the digital companion        ***")
    print("**********************************************")


### MAIN PROGRAM ###
if __name__ == '__main__':

    bot = VioletBot('sources/dev/intents.json', model_name="dev")
    bot.train_model()
    bot.save_model()
    bot.load_model()

    display_title_bar()

    done = False

    while not done:
        message = input("Enter a message: ")
        if message == "STOP" or message == 'q':
            done = True
        else:
            response = bot.request(message)
            if response is not None:
                print(response)
