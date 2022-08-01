# Откройте сайт https://parsinger.ru/infiniti_scroll_1/ с помощью Selenium;
# На сайте есть список из 100 элементов, которые генерируются при скроллинге;
# В списке есть интерактивные элементы, по которым можно осуществить скроллинг вниз;
# Цель: получить все значение в элементах, сложить их;
# Получившийся результат вставить в поле ответа.

# Подсказка:
# Элементы могут грузится медленнее чем работает ваш код, установите задержки.
# Подумайте над условием прерывания цикла, последний элемент в списке имеет class="last-of-list"



import time
from selenium.webdriver import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By


with webdriver.Chrome() as browser:
    browser.get('https://parsinger.ru/infiniti_scroll_1/')
    total = 0
    time.sleep(1)
    unique_tags = []
    scroll = False

    while not scroll:
        buttons = browser.find_element(By.ID, 'scroll-container').find_elements(By.TAG_NAME, 'span')
        for btn in buttons:
            tag_id = btn.get_attribute('id')
            if tag_id not in unique_tags:
                btn.find_element(By.TAG_NAME, 'input').send_keys(Keys.DOWN)
                total += int(btn.text)
                unique_tags.append(tag_id)
                time.sleep(0.5)
                try:
                    scroll = btn.get_attribute('class') == 'last-of-list'
                except:
                    continue
    print(total)

# 86049950