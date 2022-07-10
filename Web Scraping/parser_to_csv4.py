# Напишите код, который собирает данные в каждой категории c каждой карточки, всего их 160.
# Обязательные Заголовки :  Наименование, Артикул, Бренд, Модель, Наличие, Цена, Старая цена, Ссылка на карточку с товаром.


import requests
from bs4 import BeautifulSoup
import re
import time
import csv


class VasonParser:
    def __init__(self):
        self.categories_links = []

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
            return [add_root(tag) for tag in soup.find('div', class_='pagen').find_all('a')]
        else:
            return [add_root(tag.a) for tag in soup.find_all('div', class_='sale_button')]

    def parse_site(self, url):
        start_time = time.time()

        fields_to_parse = [
            'Наименование', 'Артикул', 'Бренд', 'Модель', 'Наличие',
            'Цена', 'Старая цена', 'Ссылка на карточку с товаром',
        ]

        with open('res.csv', 'w', encoding='utf-8-sig', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(fields_to_parse)

        self.__get_links(url)
        acc = 0

        for category in self.categories_links:
            pages_links = self.__get_links(category, categaries=False, pages=True)

            for page in pages_links:
                items_links = self.__get_links(page, categaries=False, pages=False)

                for item in items_links:
                    response = self.__get_response(item)
                    soup = self.__make_soup(response)

                    raw_descriptions = [i.text.strip().split('\n')[13:] for i in soup][1][:-1]
                    row_to_csv = []
                    indexes = [0, 1, 3, 4, -4, -2, -1]
                    for idx in indexes:
                        text = raw_descriptions[idx]
                        if ':' in text:
                            _, value = raw_descriptions[idx].split(':', 1)
                            row_to_csv.append(value.strip())
                        else:
                            row_to_csv.append(raw_descriptions[idx].strip())
                    row_to_csv.append(item)

                    acc += 1
                    with open('res.csv', 'a', encoding='utf-8-sig', newline='') as file:
                        writer = csv.writer(file, delimiter=';')
                        writer.writerow(row_to_csv)

        end_time = time.time()
        print(f'{acc} items were processed for {end_time - start_time:.2f} seconds. CSV file is ready!')



if __name__ == '__main__':
    parser = VasonParser()
    parser.parse_site('https://parsinger.ru/html/index1_page_1.html')