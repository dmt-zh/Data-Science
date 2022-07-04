# Скачайте по ссылке zip архив при помощи requests
# Извлеките из index.html его содержимое при помощи bs4 и парсера 'lxml'
# Вставьте содержимое в поле ответа

from bs4 import BeautifulSoup
import lxml
import requests
from zipfile import ZipFile
import io

zip_url = 'https://parsinger.ru/downloads/cooking_soup/index.zip'
response = requests.get(url=zip_url, stream=True)
zipped = io.BytesIO(response.content)
with ZipFile(zipped, 'r') as zip:
    lst = zip.namelist()
    for file in lst:
        if file.endswith('html'):
            with open(file, 'r', encoding='utf-8') as raw_html:
                soup = BeautifulSoup(raw_html, 'lxml')
                print(soup)