# IMPORT DISCORD.PY. ALLOWS ACCESS TO DISCORD'S API.
import json

import discord

# Import the os module.
import os

#import Violet Bot model
from violetBot import VioletBot

# GETS THE CLIENT OBJECT FROM DISCORD.PY. CLIENT IS SYNONYMOUS WITH BOT.
bot = discord.Client()
bot_name = "Violet Bot"
DISCORD_TOKEN = json.loads(open("sources/settings.json").read())['DISCORD_TOKEN']
bot_admins = ["EɢᴏʀPʟᴀʏ1™"]

# EVENT LISTENER FOR WHEN THE BOT HAS SWITCHED FROM OFFLINE TO ONLINE.
@bot.event
async def on_ready():
    # CREATES A COUNTER TO KEEP TRACK OF HOW MANY GUILDS / SERVERS THE BOT IS CONNECTED TO.
    guild_count = 0

    # LOOPS THROUGH ALL THE GUILD / SERVERS THAT THE BOT IS ASSOCIATED WITH.
    for guild in bot.guilds:
        # PRINT THE SERVER'S ID AND NAME.
        print(f"- {guild.id} (name: {guild.name})")

        # INCREMENTS THE GUILD COUNTER.
        guild_count = guild_count + 1

    # PRINTS HOW MANY GUILDS / SERVERS THE BOT IS IN.
    activity = discord.Game(name="with users on " + str(guild_count) + " servers.", type=3)
    await bot.change_presence(status=discord.Status.idle, activity=activity)
    print("Violet Bot is in " + str(guild_count) + " servers.")


# EVENT LISTENER FOR WHEN A NEW MESSAGE IS SENT TO A CHANNEL.
@bot.event
async def on_message(message):
    if message.author.name != bot_name:
        if message.content[0] == '!' and message.content[1] == 'v':
            if bot_admins.__contains__(message.author.name):
                if message.content == "!v train model":
                    violet.train_model()
                    await message.channel.send("new model trained!")
                elif message.content == "!v save model":
                    violet.save_model()
                    await message.channel.send("new model saved!")
                elif message.content == "!v load model":
                    violet.load_model()
                    await message.channel.send("new model loaded!")
                elif message.content == "!v retrain model" or message.content == "!v re-train model":
                    violet.train_model()
                    violet.save_model()
                    violet.load_model()
                    await message.channel.send("model re-trained and saved!")
                elif message.content == "!v help" or message.content == "!vhelp":
                    await message.channel.send("The bot recognizes the following commands: \n"
                                               "    **!v help** or **!vhelp** for list of valid commands\n"
                                               "    **!v train model** to train a new model for ai\n"
                                               "    **!v save model** to save the new model for ai\n"
                                               "    **!v load model** to load the last saved model for ai\n"
                                               "    **!v retrain model** or **!v re-train model** to train and save a "
                                               "new model and then apply it for ai\n")
                elif message.content == "!v stop" or  message.content == "!vstop":
                    await message.channel.send("going to bed")
                    exit()
                else:
                    result = violet.request(message.content)
                    await message.channel.send(result)
            else:
                await message.channel.send("You are not an admin to use internal commands, contact @EɢᴏʀPʟᴀʏ1™#5700 "
                                           "if you believe this is an error!")
        else:
            # SENDS BACK A MESSAGE TO THE CHANNEL.
            result = violet.request(message.author.name, message.content)
            if result != "":
                await message.channel.send(result)


# EXECUTES THE BOT WITH THE SPECIFIED TOKEN.
if __name__ == '__main__':
    violet = VioletBot('sources/dev/intents.json', model_name="dev")
    # violet.train_model()
    # violet.save_model()
    violet.load_model()
    bot.run(DISCORD_TOKEN)
