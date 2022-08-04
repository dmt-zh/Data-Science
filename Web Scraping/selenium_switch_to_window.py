# Откройте сайт https://parsinger.ru/blank/3/index.html с помощью Selenium;
# На сайте есть 10 buttons, каждый button откроет сайт в новой вкладке;
# Каждая вкладка имеет в title уникальное число;
# Цель - собрать числа с каждой вкладки и суммировать их;
# Полученный результат вставить в поле для ответа.



from selenium import webdriver
from selenium.webdriver.common.by import By

total = 0
with webdriver.Chrome() as browser:
    browser.get('https://parsinger.ru/blank/3/index.html')
    for button in browser.find_elements(By.CLASS_NAME, 'buttons'):
        button.click()

    for w in browser.window_handles[1:]:
        browser.switch_to.window(w)
        title = browser.execute_script("return document.title;")
        total += int(title)

print(total)
# 77725787998028643152187739088279
