# Цель: собрать username всех пользователей которые отправили числовое сообщение в группу;
# Создать список из этих username;
# Вставить полученный список в поле для ответа.

# Ожидаемый список (символ @ добавлять к username не нужно), в списке может быть только 1 username пользователя.
# Если у пользователя отсутствует username, исключить его из списка: [username, username, ..., username]


from telethon import TelegramClient, events, sync, connection
from tg_connections import tg_api, tg_hash


with TelegramClient('tgparser', tg_api, tg_hash) as client:
    messages = client.iter_messages('https://t.me/Parsinger_Telethon_Test')
    check_if_digit = lambda txt: True if txt.message is not None and txt.message.isdigit() else False
    get_user_name = lambda user: client.get_entity(user.from_id.user_id).username
    unique_users = set(map(get_user_name, filter(check_if_digit, messages)))
    unique_users.discard(None)
    print(list(unique_users))


# ['Mary_Sanchez324', 'Nancy_Montgomery54', 'Lisa_Hawkins', 'Cindy_Porter', 'Mark_Mendez980', 'Richard_Welch',
# 'Brian_Johnson2', 'John_Harris43', 'George_Webster43', 'dogym0onso', 'Mildred_James', 'James_Washington3',
# 'Robert_Jones34', 'William_Price34', 'Nancy_Johnson3', 'Thomas_Jones56', 'Elizabeth_Weber', 'Erica_Moore34',
#  'Linda_Hernandez4', 'Nathan_King43', 'Scott_Stevenson32', 'Sara_Martin434', 'Gloria_Thompson4', 'Joshua_Andrews34']