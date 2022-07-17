# Выберите 1 любую категорию на сайте тренажёре, и соберите все данные с карточек товаров + ссылка на карточку.
# По результату выполнения кода в папке с проектом должен появится файл .json с отступом в 4 пробела.
# Ключи в блоке description должны быть получены автоматически из атрибутов HTML элементов.



import requests
from bs4 import BeautifulSoup
import re
import time
import json


class HDDParser2:
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

    def __get_links(self, url, categaries=True, pages=False):
        root = re.match(r'^(http|https)://[a-z.]+/[a-z]+/', url).group(0)
        response = self.__get_response(url)
        soup = self.__make_soup(response)
        add_root = lambda x: root + x['href']
        if categaries:
            for tag in soup.find('div', class_='nav_menu').find_all('a'):
                self.categories_links.append(add_root(tag))
        if pages:
            return list(map(add_root, soup.find('div', class_='pagen').find_all('a')))
        else:
            return list(add_root(tag.a) for tag in soup.find_all('div', class_='sale_button'))

    def parse_site(self, url):
        start_time = time.time()
        print('Scraping is started.')
        acc = 0
        pages_links = self.__get_links(url, categaries=False, pages=True)

        for page in pages_links:
            items_links = self.__get_links(page, categaries=False, pages=False)

            for item_link in items_links:
                soup = self.__make_soup(self.__get_response(item_link))

                item = {}
                description = {}

                category = 'HDD'
                name = soup.find('p', id='p_header').text.strip()
                article = soup.find('p', class_='article').text.split(': ')[-1]

                for k, v in zip(('categories', 'name', 'article'), (category, name, article)):
                    item[k] = v

                all_li = soup.find('ul', id='description').find_all('li')
                attr_descr = (i['id'].strip() for i in all_li)
                vals_descr = (i.text.split(':', 1)[-1].strip() for i in all_li)

                for k, v in zip(attr_descr, vals_descr):
                    description[k] = v
                item['description'] = description

                attr_item = (i['id'] for i in soup.find_all('span'))
                desc_item = (i.text.split(':')[-1].strip() for i in soup.find_all('span'))

                for k, v in zip(attr_item, desc_item):
                    item[k] = v
                item['link'] = item_link

                self.result_json.append(item)
                description = {}
                item = {}
                acc += 1

        with open('res.json', 'w', encoding='utf-8') as file:
            json.dump(self.result_json, file, indent=4, ensure_ascii=False)

        end_time = time.time()
        print(f'{acc} items were processed for {end_time - start_time:.2f} seconds. JSON file is ready!')



if __name__ == '__main__':
    parser = HDDParser2()
    parser.parse_site('https://parsinger.ru/html/index4_page_1.html')