# Соберите данные со всех 5 категорий на сайте тренажере и соберите все данные с карточек.
# По результату выполнения кода в папке с проектом должен появится файл .json с отступом в 4 пробела.
# Ключи в блоке description должны быть получены автоматически из атрибутов HTML элементов.



import requests
from bs4 import BeautifulSoup
import re
import time
import json



class GoodsParser:
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
        soup = self.__make_soup(self.__get_response(url))
        add_root = lambda x: root + x['href']
        if categaries:
            for tag in soup.find('div', class_='nav_menu').find_all('a'):
                self.categories_links.append(add_root(tag))
        if pages:
            return map(add_root, soup.find('div', class_='pagen').find_all('a'))
        else:
            return (add_root(tag.a) for tag in soup.find_all('div', class_='sale_button'))

    def parse_site(self, url):
        print('Scraping is started.')
        start_time = time.time()
        acc = 0

        self.__get_links(url)
        c_soup = self.__make_soup(self.__get_response(url))
        category_names = map(lambda x: x['id'], c_soup.find('div', class_='nav_menu').find_all('div'))

        for category_link, category in zip(self.categories_links, category_names):
            pages_links = self.__get_links(category_link, categaries=False, pages=True)

            for page in pages_links:
                items_links = self.__get_links(page, categaries=False, pages=False)

                for item_link in items_links:
                    soup = self.__make_soup(self.__get_response(item_link))

                    item = {}
                    description = {}

                    name = soup.find('p', id='p_header').text.strip()
                    article = soup.find('p', class_='article').text.split(': ')[-1]
                    for k, v in zip(('categories', 'name', 'article'), (category, name, article)):
                        item[k] = v

                    attr_descr = map(lambda x: x['id'].strip(), soup.find('ul', id='description').find_all('li'))
                    vals_descr = map(lambda x: x.text.split(':', 1)[-1].strip(), soup.find('ul', id='description').find_all('li'))

                    for k, v in zip(attr_descr, vals_descr):
                        if category == 'mouse' and k == 'purpose':
                            description[k] = 'Игровая: ' + v
                        else:
                            description[k] = v
                    item['description'] = description

                    attr_item = map(lambda x: x['id'], soup.find_all('span'))
                    vals_item = map(lambda x: x.text.split(':')[-1].strip(), soup.find_all('span'))

                    for k, v in zip(attr_item, vals_item):
                        if k.startswith('in_stock'):
                            item['count'] = v
                        else:
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
    parser = GoodsParser()
    parser.parse_site('https://parsinger.ru/html/index1_page_1.html')