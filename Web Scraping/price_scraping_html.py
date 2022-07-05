# Открываем сайт https://parsinger.ru/html/index1_page_1.html
# Извлекаем при помощи bs4 данные о стоимости часов (всего 8 шт)
# Складываем все числа
# Вставляем результат в поле ответа



from bs4 import BeautifulSoup
import requests

url = 'https://parsinger.ru/html/index1_page_1.html'
response = requests.get(url=url)
response.encoding = 'utf-8'
soup = BeautifulSoup(response.text, 'lxml')
print(sum(map(lambda x: int(x.text.split()[0]), soup.find_all('p', class_='price'))))