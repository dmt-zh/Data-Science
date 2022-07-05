# Открываем сайт https://parsinger.ru/html/hdd/4/4_1.html
# Получаем данные при помощи bs4 о старой цене и новой цене
# По формуле высчитываем процент скидки
# Формула (старая цена - новая цена) * 100 / старая цена)
# Вставьте получившийся результат в поле ответа
# Ответ должен быть числом с 1 знаком после запятой.



from bs4 import BeautifulSoup
import requests

def get_discount(previous, current):
    discount = round((previous - current) * 100 / previous, 1)
    return discount

url = 'https://parsinger.ru/html/hdd/4/4_1.html'
response = requests.get(url=url)
response.encoding = 'utf-8'
soup = BeautifulSoup(response.text, 'lxml')
price = int(soup.find('span', id='price').text.split()[0])
old_price = int(soup.find('span', id='old_price').text.split()[0])
print(get_discount(old_price, price))