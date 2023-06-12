# IMPORT DISCORD.PY. ALLOWS ACCESS TO DISCORD'S API.
import io
import json
import math
import sys

import discord
from datetime import datetime
import time
import logging

# Import the os module.
import os

from v2.violetBot import VioletBot

# GETS THE CLIENT OBJECT FROM DISCORD.PY. CLIENT IS SYNONYMOUS WITH BOT.
bot = discord.Client(intents=discord.Intents.default())
bot_name = "Violet"
DISCORD_TOKEN = json.loads(open("../sources/settings.json").read())['DISCORD_TOKEN']
bot_admins = ["EɢᴏʀPʟᴀʏ1™", "WHATISLOVE", "EgorPlay1"]


# EVENT LISTENER FOR WHEN THE BOT HAS SWITCHED FROM OFFLINE TO ONLINE.
@bot.event
async def on_ready():
    logging.info(f'Starting Violet Bot at {datetime.now()}')
    # CREATES A COUNTER TO KEEP TRACK OF HOW MANY GUILDS / SERVERS THE BOT IS CONNECTED TO.
    guild_count = 0

    # LOOPS THROUGH ALL THE GUILD / SERVERS THAT THE BOT IS ASSOCIATED WITH.
    for guild in bot.guilds:
        # PRINT THE SERVER'S ID AND NAME.
        logging.debug(f'- {guild.id} (name: {guild.name})')

        # INCREMENTS THE GUILD COUNTER.
        guild_count = guild_count + 1

    # PRINTS HOW MANY GUILDS / SERVERS THE BOT IS IN.
    activity = discord.Game(name="with users on " + str(guild_count) + " servers.", type=3)
    await bot.change_presence(status=discord.Status.idle, activity=activity)
    logging.info(f'Violet Bot is in {str(guild_count)} servers.\n')


# EVENT LISTENER FOR WHEN A NEW MESSAGE IS SENT TO A CHANNEL.
@bot.event
async def on_message(message: discord.Message) -> None:
    logging.debug(f'Received message from {message.author.name}: {message.content}')
    if message.author.name != bot_name and isinstance(message.channel, discord.DMChannel):
        await message.channel.send(violet_bot.get_response(message.content))


# EXECUTES THE BOT WITH THE SPECIFIED TOKEN.
if __name__ == '__main__':
    logging.basicConfig(filename=f'example{datetime.now().date()}.log', encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    violet_bot = VioletBot()
    bot.run(DISCORD_TOKEN)
