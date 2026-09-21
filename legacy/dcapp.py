import os
from dotenv import load_dotenv
load_dotenv()
import discord
from discord.ext import commands
from appV1 import retriever_qa   # <-- clean import, CLI loop does NOT run

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user in message.mentions:
        question = message.content.replace(f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '').strip()
        if not question:
            await message.channel.send("Ask me something!")
            return

        answer = retriever_qa(question)
        await message.channel.send(answer)

    await bot.process_commands(message)

bot.run(os.getenv("DC_API_KEY"))