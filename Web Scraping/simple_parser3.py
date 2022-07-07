# Открываем сайт 'https://parsinger.ru/html/index1_page_1.html'
# Проходимся по всем категориям, страницам и карточкам с товарами(всего 160 шт)
# Собираем с каждой карточки стоимость товара умножая на количество товара в наличии
# Складываем получившийся результат
# Получившуюся цифру с общей стоимостью всех товаров вставляем в поле ответа.



from bs4 import BeautifulSoup
import requests
import numpy as np


class VazonParser:
    def __init__(self):
        self.prices = []
        self.in_stock = []

    @staticmethod
    def __get_response(url):
        response = requests.get(url=url)
        response.encoding = 'utf-8'
        return response

    @staticmethod
    def __make_soup(response):
            return BeautifulSoup(response.text, 'lxml')

    def __find_all_links(self, response, root, class_name, add_find=False):
        soup = self.__make_soup(response)
        if add_find:
            return map(lambda x: root + x.a['href'], soup.find_all('div', class_=class_name))
        else:
            return map(lambda x: root + x['href'], soup.find('div', class_=class_name).find_all('a'))

    @staticmethod
    def __find_total_sum(price, count):
        return np.sum(np.array(price) * np.array(count))

    def parse_site(self, url, root):
        categories_links = self.__find_all_links(self.__get_response(url), root, 'nav_menu')

        for category in categories_links:
            pages_links = self.__find_all_links(self.__get_response(category), root, 'pagen')

            for page in pages_links:
                items_links = self.__find_all_links(self.__get_response(page), root, 'sale_button', add_find=True)

                for i_link in items_links:
                    soup = self.__make_soup(self.__get_response(i_link))
                    price = soup.find('span', id='price').text.split()[0]
                    quantity = soup.find('span', id='in_stock').text.split()[-1]
                    self.prices.append(int(price))
                    self.in_stock.append(int(quantity))

        return self.__find_total_sum(self.prices, self.in_stock)


if __name__ == '__main__':
    url = 'https://parsinger.ru/html/index1_page_1.html'
    site = 'http://parsinger.ru/html/'
    parser = VazonParser()
    print(parser.parse_site(url, site))