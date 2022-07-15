# Выберите 1 любую категорию на сайте тренажере и соберите все данные с карточек.
# По результату выполнения кода в папке с проектом должен появится файл .json с отступом в 4 пробела.


import requests
from bs4 import BeautifulSoup
import re
import time
import json


class PhoneParser:
    def __init__(self):
        self.result_json = []

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
            return (add_root(tag) for tag in soup.find('div', class_='pagen').find_all('a'))
        else:
            return (add_root(tag.a) for tag in soup.find_all('div', class_='sale_button'))

    def parse_site(self, url):
        start_time = time.time()
        pages_links = self.__get_links(url)
        acc = 0

        for page in pages_links:
            response = self.__get_response(page)
            soup = self.__make_soup(response)

            features = ('name', 'brand', 'diagonal', 'material', 'resolution', 'price')
            item = {}
            descriptions = ([txt.strip() for txt in i.text.split('\n') if txt != ''][:-1] for i in soup.find_all('div', class_='item'))

            for nested in descriptions:
                for k, v in zip(features, nested):
                    if ':' in v:
                        _, value = v.split(':')
                        item[k] = value.strip()
                    else:
                        item[k] = v
                self.result_json.append(item)
                item = {}
                acc += 1

        with open('res.json', 'w', encoding='utf-8') as file:
            json.dump(self.result_json, file, indent=4, ensure_ascii=False)

        end_time = time.time()
        print(f'{acc} items were processed for {end_time - start_time:.2f} seconds. JSON file is ready!')


if __name__ == '__main__':
    parser = PhoneParser()
    parser.parse_site('https://parsinger.ru/html/index2_page_1.html')