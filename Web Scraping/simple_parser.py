# Откройте сайт https://parsinger.ru/html/index3_page_1.html
# Извлеките названия товара с каждой страницы (всего 4х страниц)
# Данные с каждой страницы должны хранится в списке.
# По итогу работы должны получится 4 списка которые хранятся в списке(список списков)
# Отправьте получившийся список списков в поле ответа.
# Метод strip()использовать не нужно

from bs4 import BeautifulSoup
import requests

class VazonParser:
    def __init__(self):
        self.descriptions = []

    def __get_response(self, url):
        response = requests.get(url=url)
        response.encoding = 'utf-8'
        return response

    def __find_pages(self, response, site):
        soup = BeautifulSoup(response.text, 'lxml')
        pagen = map(lambda x: site + x['href'], soup.find('div', class_='pagen').find_all('a'))
        return pagen

    def parse_site(self, url, site):
        response = self.__get_response(url)
        for page in self.__find_pages(response, site):
            soup = BeautifulSoup(self.__get_response(page).text, 'lxml')
            raw_text = soup.find_all('a', class_='name_item')
            description = list(map(lambda x: x.text, raw_text))
            self.descriptions.append(description)
        print(self.descriptions)


if __name__ == '__main__':
    url = 'https://parsinger.ru/html/index3_page_1.html'
    site = 'http://parsinger.ru/html/'
    parser = VazonParser()
    parser.parse_site(url, site)