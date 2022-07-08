# На  сайте расположена таблица;
# Цель: Умножить число в оранжевой ячейке на число в голубой ячейке в той же строке и всё суммировать;
# Полученный результат вставить в поле ответа.



from bs4 import BeautifulSoup
import requests
import numpy as np


url = "https://parsinger.ru/table/5/index.html"
response = requests.get(url=url)
soup = BeautifulSoup(response.text, 'lxml')

blue_cells = [int(x.find_all('td')[-1].text) for x in soup.find_all('tr') if x.find('td')]
orange_cells = [float(cell.text) for cell in soup.find_all('td', class_='orange')]

print(np.array(orange_cells) @ np.array(blue_cells))