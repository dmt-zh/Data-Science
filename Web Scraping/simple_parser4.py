# Напишите код, который собирает данные в категории watch c каждой карточки, всего их 32.
# Обязательные Заголовки :  Наименование, Артикул, Бренд, Модель, Тип, Технология экрана,
# Материал корпуса, Материал браслета, Размер, Сайт производителя, Наличие, Цена, Старая цена,  Ссылка на карточку с товаром.
# Всего должно быть 14 заголовков


import requests
from bs4 import BeautifulSoup
import re
import time
import csv


class WatchParser:
    def __init__(self):
        self.pages_links = []

    @staticmethod
    def __get_response(url):
        response = requests.get(url=url)
        response.encoding = 'utf-8'
        return response

    @staticmethod
    def __make_soup(response):
        return BeautifulSoup(response.text, 'lxml')

    def __get_links(self, url, pages=True):
        root = re.match(r'^(http|https)://[a-z.]+/[a-z]+/', url).group(0)
        response = self.__get_response(url)
        soup = self.__make_soup(response)
        add_root = lambda x: root + x['href']
        if pages:
            raw_pages_tags = soup.find('div', class_='pagen').find_all('a')
            for tag in raw_pages_tags:
                self.pages_links.append(add_root(tag))
        else:
            raw_items_tags = soup.find_all('div', class_='sale_button')
            return [add_root(tag.a) for tag in raw_items_tags]

    def parse_site(self, url):
        start_time = time.time()
        self.__get_links(url)

        fields_to_parse = [
            'Наименование', 'Артикул', 'Бренд', 'Модель', 'Тип', 'Технология экрана',
            'Материал корпуса', 'Материал браслета', 'Размеры', 'Сайт производителя',
            'Наличие', 'Цена', 'Старая цена', 'Ссылка на карточку с товаром',
        ]

        with open('res.csv', 'w', encoding='utf-8-sig', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(fields_to_parse)

        for page in self.pages_links:
            items_links = self.__get_links(page, pages=False)

            for item in items_links:
                response = self.__get_response(item)
                soup = self.__make_soup(response)

                brand = soup.find('p', id='p_header').text.strip()
                article = soup.find('p', class_='article').text.split(':')[-1].strip()

                raw_descriptions = [i.text.split('\n')[1:-1] for i in soup.find('div', class_='description')]
                part_row = []
                for text in raw_descriptions[5]:
                    _, value = text.split(':', 1)
                    part_row.append(value.strip())

                in_stock = soup.find('span', id='in_stock').text.split(':')[-1].strip()
                price = raw_descriptions[9][0]
                old_price = raw_descriptions[9][1]

                total_row = [brand, article, *part_row, in_stock, price, old_price, item]
                with open('res.csv', 'a', encoding='utf-8-sig', newline='') as file:
                    writer = csv.writer(file, delimiter=';')
                    writer.writerow(total_row)

        end_time = time.time()
        print(f'The process has taken {end_time - start_time:.2f} seconds. CSV file is ready!')



if __name__ == '__main__':
    parser = WatchParser()
    parser.parse_site('https://parsinger.ru/html/index1_page_1.html')