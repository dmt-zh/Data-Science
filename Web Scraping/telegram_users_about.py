# Есть список lst=[] в котором хранятся username участников группы;
# Цель собрать числа из поля "О Себе" или "About" пользователя из списка lst=[], затем суммировать все добытые числа;
# Полученное число вставить в поле для ответа.


from telethon import TelegramClient, events, sync, connection
from telethon.tl.functions.users import GetFullUserRequest
from tg_connections import tg_api, tg_hash


lst = ['daxton_13246', 'Anthony_Alexander534', 'William_Price34', 'Roger_Parks4', 'Nancy_Montgomery54',
       'Melissa_Simmons4', 'Shane_Morris34', 'Gloria_Thompson4', 'Linda_Hernandez4',
       'Constance_Jones4', 'Joshua_Andrews34', 'Erica_Moore34', 'Timothy_Green3', 'Lisa_Hawkins',
       'Nancy_Johnson3', 'Mary_Davis1', 'Brian_Johnson2', 'Peter_Barnes', 'James_Washington3']

res = 0
with TelegramClient('tgparser', tg_api, tg_hash) as client:
    users = client.iter_participants('https://t.me/Parsinger_Telethon_Test')
    usernames = [u.username for u in users]
    print(usernames)
    for user in users:
        if user.username in lst:
            user_full = client(GetFullUserRequest(user))
            try:
                res += int(user_full.about)
            except:
                continue
print(res)
