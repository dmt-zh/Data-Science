# На  сайте расположена таблица;
# Цель: Собрать числа которые выделены жирным шрифтом и суммировать их;
# Полученный результат вставить в поле ответа.



from bs4 import BeautifulSoup
import requests

url = "https://parsinger.ru/table/3/index.html"
response = requests.get(url=url)
soup = BeautifulSoup(response.text, 'lxml')
table = soup.find_all('b')
print(sum([float(t.text) for t in table]))