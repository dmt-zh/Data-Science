# В группе есть закреплённое сообщение;
# Цель: получить user_id пользователя чьё сообщение закреплено;
# Вставить полученный user_id в поле для ответа.


from telethon import TelegramClient, events, sync, connection
from telethon.tl.types import InputMessagesFilterPinned
from tg_connections import tg_api, tg_hash

with TelegramClient('tgparser', tg_api, tg_hash) as client:
    message = client.get_messages('https://t.me/Parsinger_Telethon_Test', filter=InputMessagesFilterPinned)
    pinned_user_id = tuple(map(lambda x: x.from_id.user_id, message))[0]
    print(pinned_user_id)

# 5330282124