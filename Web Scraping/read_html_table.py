# На  сайте расположена таблица https://parsinger.ru/table/1/index.html;
# Цель: Собрать все уникальные числа из таблицы(кроме цифр в заголовке) и суммировать их;
# Полученный результат вставить в поле ответа.


from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


# С помощью BeautifulSoup
url = "https://parsinger.ru/table/1/index.html"
response = requests.get(url=url)
response.encoding = 'utf-8'
soup = BeautifulSoup(response.text, 'lxml')
table = soup.find_all('td')
print(sum(set([float(t.text) for t in table])))


#  С помощью Pandas + Numpy
data = pd.read_html('https://parsinger.ru/table/1/index.html', header=0)
df = data[0]
print(np.unique(pd.melt(df)['value'].values).sum())