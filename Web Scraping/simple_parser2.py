# Открываем сайт 'https://parsinger.ru/html/index3_page_1.html'
# Проходимся по всем страницам в категории мыши (всего  4 страницы)
# На каждой странице посещаем каждую карточку с товаром (всего 32 товаров)
# В каждой карточке извлекаем при помощи bs4 артикул <p class="article"> Артикул: 80244813 </p>
# Складываем(плюсуем) все собранные значения
# Вставляем получившийся результат в поле ответа



from bs4 import BeautifulSoup, SoupStrainer
import requests


class MouseParser:
    def __init__(self):
        self.articules = []

    def __get_response(self, url):
        response = requests.get(url=url)
        response.encoding = 'utf-8'
        return response

    def __make_soup(self, response, links_only=False):
        if links_only:
            return BeautifulSoup(response.text, 'html.parser', parse_only=SoupStrainer('a'))
        else:
            return BeautifulSoup(response.text, 'html.parser')

    def parse_site(self, url, root):
        response = self.__get_response(url)
        for page in self.__make_soup(response).find('div', class_='pagen').find_all('a'):
            p_link = root + page['href']
            p_response = self.__get_response(p_link)
            p_soup = self.__make_soup(p_response, links_only=True)

            items_links = set(root + tag['href'] for tag in p_soup if tag['href'].startswith('mouse'))
            for i_link in items_links:
                i_response = self.__get_response(i_link)
                article = self.__make_soup(i_response).find('p', class_='article').text.split()[-1]
                self.articules.append(article)

        return sum(map(int, self.articules))


if __name__ == '__main__':
    url = 'https://parsinger.ru/html/index3_page_1.html'
    site = 'http://parsinger.ru/html/'
    parser = MouseParser()
    print(parser.parse_site(url, site))