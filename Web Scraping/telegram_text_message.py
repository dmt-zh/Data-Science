# Спарсить числовые значения из сообщений в группе;
# Суммировать полученные числа и вставить результат в поле для ответа.


from telethon import TelegramClient, events, sync, connection
from tg_connections import tg_api, tg_hash

total = 0
with TelegramClient('tgparser', tg_api, tg_hash) as client:
    all_messages = client.iter_messages('https://t.me/Parsinger_Telethon_Test')
    for msg in all_messages:
        text = msg.message
        if text is not None and text.isdigit():
            total += int(text)

print(total)