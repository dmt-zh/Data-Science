# Откройте https://parsinger.ru/window_size/2/index.html сайт с помощью selenium;
# У вас есть 2 списка с размера окон size_x и size_y;
# Цель: определить размер окна, при котором,  в id="result" появляется число;
# Результат должен быть в виде словаря {'width': size_x, 'height': size_y}


from selenium import webdriver
from selenium.webdriver.common.by import By
from itertools import product


with webdriver.Chrome() as browser:
    browser.get('https://parsinger.ru/window_size/2/index.html')

    window_size_x = [484, 516, 648, 680, 701, 730, 750, 805, 820, 855, 890, 955, 1000]
    window_size_y = [250, 270, 300, 340, 388, 400, 421, 474, 505, 557, 600, 653, 1000]

    for x, y in product(window_size_x, window_size_y):
        browser.set_window_size(x+16, y+92)
        result = browser.find_element(By.ID, 'result').text
        if result:
            print(result)
            print(browser.get_window_size())
            break



# 9874163854135461654
# {'width': 955, 'height': 600}