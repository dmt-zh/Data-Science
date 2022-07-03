# Откройте сайт https://parsinger.ru/task/1/
# На нём есть 500 ссылок  и только 1 вернёт статус код 200
# Напишите код который поможет найти правильную ссылку
# По этой ссылке лежит секретный код, который необходимо вставить в поле ответа.


import requests
from bs4 import BeautifulSoup

def get_valid_link():
    for number in range(1, 501):
        url = f'https://parsinger.ru/task/1/{number}.html'
        response = requests.get(url=url)
        if response.status_code == 200:
            text = response.text
            soup = BeautifulSoup(text, 'lxml')
            secret = soup.body.text.strip()
            print(f'Valid link is "{number}.html" \nThe secret code: {secret}')
            break
        else:
            continue

if __name__ == '__main__':
    link = get_valid_link()