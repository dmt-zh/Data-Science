# Напишите код, который собирает данные в категории HDD со всех 4х страниц и сохраняет всё в таблицу CSV.


import requests
from bs4 import BeautifulSoup
import re
import pandas as pd


class HddParser:
    def __init__(self):
        self.all_data = {'Наименование': [], 'Цена': [], 'Бренд': [], 'Форм-фактор': [], 'Ёмкость': [], 'Объем буферной памяти': []}
        self.pages_links = []

    @staticmethod
    def __get_response(url):
        response = requests.get(url=url)
        response.encoding = 'utf-8'
        return response

    @staticmethod
    def __make_soup(response):
        return BeautifulSoup(response.text, 'lxml')

    def __get_pages_links(self, url):
        root = re.match(r'^(http|https)://[a-z.]+/[a-z]+/', url).group(0)
        response = self.__get_response(url)
        soup = self.__make_soup(response).find('div', class_='pagen').find_all('a')
        for tag in soup:
            link = root + tag['href']
            self.pages_links.append(link)

    def parse_site(self, url):
        self.__get_pages_links(url)

        for page in self.pages_links:
            response = self.__get_response(page)
            soup = self.__make_soup(response)

            for name in soup.find_all('a', class_='name_item'):
                brand_name = name.text.strip()
                self.all_data['Наименование'].append(brand_name)

            for price in soup.find_all('p', class_='price'):
                item_price = price.text
                self.all_data['Цена'].append(item_price)

            raw_descriptions = [x.text.split('\n')[1:-1] for x in soup.find_all('div', class_='description')]
            for nested in raw_descriptions:
                for description in nested:
                    key, value = description.split(':')
                    self.all_data[key].append(value.strip())

        df = pd.DataFrame(self.all_data)
        df.to_csv('res.csv', sep=';', index=False, encoding='utf-8-sig')
        print('The process is finished. CSV file is ready!')


if __name__ == '__main__':
    url = 'https://parsinger.ru/html/index4_page_1.html'
    parser = HddParser()
    parser.parse_site(url)