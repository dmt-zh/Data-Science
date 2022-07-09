# Напишите код, который собирает данные со всех страниц и категорий на сайте тренажере и сохраните всё в таблицу.
# Заголовки :  Указывать не нужно

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd


class VazonParser:
    def __init__(self):
        self.all_data = {'column_1': [], 'column_2': [], 'column_3': [], 'column_4': [], 'column_5': [], 'column_6': []}
        self.categories_links = []
        self.pages_links = []

    @staticmethod
    def __get_response(url):
        response = requests.get(url=url)
        response.encoding = 'utf-8'
        return response

    @staticmethod
    def __make_soup(response):
        return BeautifulSoup(response.text, 'lxml')

    def __get_links(self, url, category=True):
        root = re.match(r'^(http|https)://[a-z.]+/[a-z]+/', url).group(0)
        response = self.__get_response(url)
        soup = self.__make_soup(response)
        if category:
            raw_navigation_tags = soup.find('div', class_='nav_menu').find_all('a')
            for tag in raw_navigation_tags:
                link = root + tag['href']
                self.categories_links.append(link)
        else:
            raw_pages_tags = soup.find('div', class_='pagen').find_all('a')
            for tag in raw_pages_tags:
                link = root + tag['href']
                self.pages_links.append(link)

    def parse_site(self, url):
        self.__get_links(url)
        for category in self.categories_links:
            self.__get_links(category, category=False)

        for page in self.pages_links:
            response = self.__get_response(page)
            soup = self.__make_soup(response)

            for name in soup.find_all('a', class_='name_item'):
                brand_name = name.text.strip()
                self.all_data['column_1'].append(brand_name)

            for price in soup.find_all('p', class_='price'):
                item_price = price.text
                self.all_data['column_6'].append(item_price)

            raw_descriptions = map(lambda x: x.text.split('\n')[1:-1], soup.find_all('div', class_='description'))
            for nested in raw_descriptions:
                for n, description in enumerate(nested, 2):
                    _, value = description.split(':')
                    self.all_data[f'column_{n}'].append(value.strip())

        df = pd.DataFrame(self.all_data)
        df.to_csv('res.csv', sep=';', index=False, encoding='utf-8-sig', header=False)
        print('The process is finished. CSV file is ready!')



if __name__ == '__main__':
    url = 'https://parsinger.ru/html/index1_page_1.html'
    parser = VazonParser()
    parser.parse_site(url)