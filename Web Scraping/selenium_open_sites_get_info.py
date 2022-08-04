# У вас есть список сайтов, 6 шт;
# На каждом сайте есть chekbox, нажав на этот chekbox появится код;
# Ваша задача написать скрипт, который открывает при помощи Selenium все сайты во вкладках;
# Проходит в цикле по каждой вкладке, нажимает на chekbox и сохранеят код;
# Из каждого числа, необходимо извлечь корень, функцией sqrt();
# Суммировать получившиеся корни и вставить результат в поле для ответа.


from selenium import webdriver
from selenium.webdriver.common.by import By

sites = [
    'http://parsinger.ru/blank/1/1.html', 'http://parsinger.ru/blank/1/2.html', 'http://parsinger.ru/blank/1/3.html',
    'http://parsinger.ru/blank/1/4.html', 'http://parsinger.ru/blank/1/5.html', 'http://parsinger.ru/blank/1/6.html'
]


codes = []
with webdriver.Chrome() as browser:
    for site in sites:
        browser.get(site)
        check_box = browser.find_element(By.CLASS_NAME, 'checkbox_class')
        check_box.click()
        result = browser.find_element(By.ID, 'result').text
        codes.append(result)

    print(round(sum(map(lambda x: int(x) ** .5, codes)), 9))

# 334703.720482347