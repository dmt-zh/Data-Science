# Соберите данные со всех 5 категорий на сайте тренажере и соберите все данные с карточек.
# По результату выполнения кода в папке с проектом должен появится файл .json с отступом в 4 пробела.


import requests
from bs4 import BeautifulSoup
import re
import time
import json


class VazonParser2:
    def __init__(self):
        self.categories_links = []
        self.result_json = []

    @staticmethod
    def __get_response(url):
        response = requests.get(url=url)
        response.encoding = 'utf-8'
        return response

    @staticmethod
    def __make_soup(response):
        return BeautifulSoup(response.text, 'lxml')

    def __get_links(self, url, categaries=True, pages=False):
        root = re.match(r'^(http|https)://[a-z.]+/[a-z]+/', url).group(0)
        response = self.__get_response(url)
        soup = self.__make_soup(response)
        add_root = lambda x: root + x['href']
        if categaries:
            for tag in soup.find('div', class_='nav_menu').find_all('a'):
                self.categories_links.append(add_root(tag))
        if pages:
            return map(add_root, soup.find('div', class_='pagen').find_all('a'))
        else:
            return add_root(soup.find('div', class_='sale_button').a)

    def parse_site(self, url):
        start_time = time.time()
        self.__get_links(url)
        acc = 0

        for category in self.categories_links:
            pages_links = self.__get_links(category, categaries=False, pages=True)

            for page in pages_links:
                soup = self.__make_soup(self.__get_response(page))
                item = {}

                descriptions = (tuple(t.strip().split(':') for t in i.text.split('\n') if t != '' or ':' in t)[:-1] \
                                for i in soup.find_all('div', class_='item'))

                attrs = {'Бренд': 'brand', 'Тип': 'type', 'Материал корпуса': 'material_frame', 'Технология экрана': 'display',
                         'Диагональ экрана': 'diagonal', 'Разрешение экрана': 'resolution', 'Подключение к компьютеру': 'connect',
                         'Игровая': 'game', 'Форм-фактор': 'form-factor', 'Ёмкость': 'capacity', 'Объем буферной памяти': 'buffer-memory',
                         'Тип подключения': 'connect', 'Тип наушников': 'type', 'Цвет': 'color'}

                for nested in descriptions:
                    item['name'] = nested[0][0].strip()
                    for desc in nested[1:-1]:
                        key, value = desc
                        item[attrs.get(key.strip())] = value.strip()
                    item['price'] = nested[-1][0]
                    self.result_json.append(item)
                    if len(item) != 6:
                        print(item, page)
                    item = {}
                    acc += 1

        with open('res.json', 'w', encoding='utf-8') as file:
            json.dump(self.result_json, file, indent=4, ensure_ascii=False)

        end_time = time.time()
        print(f'{acc} items were processed for {end_time - start_time:.2f} seconds. JSON file is ready!')



if __name__ == '__main__':
    parser = VazonParser2()
    parser.parse_site('https://parsinger.ru/html/index1_page_1.html')