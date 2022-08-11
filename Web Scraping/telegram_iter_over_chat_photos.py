# Скачайте все изображения из группы;
# Определите общий размер всех фотографий(40шт);
# Вставьте полученное число в поле для ответа(число должно быть в байтах, в числе должны отсутствовать пробелы).



from telethon import TelegramClient, events, sync, connection
from telethon.tl.types import InputMessagesFilterPhotos
from tg_connections import tg_api, tg_hash
import os

size = 0
with TelegramClient('tgparser', tg_api, tg_hash) as client:
    for message in client.iter_messages('https://t.me/Parsinger_Telethon_Test', filter=InputMessagesFilterPhotos):
        file_name = f'img\{message.id}.jpg'
        client.download_media(message, file=file_name)
        size += os.path.getsize(file_name)

print(size)
# 1256602