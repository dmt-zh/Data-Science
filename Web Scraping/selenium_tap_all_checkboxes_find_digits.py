# Откройте сайт https://parsinger.ru/scroll/2/index.html с помощью Selenium;
# На сайте есть 100 чекбоксов, 25 из них вернут число;
# Ваша задача суммировать все появившиеся числа;
# Отправить получившийся результат в поля ответа.



from selenium.webdriver import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By

options_chrome = webdriver.ChromeOptions()
options_chrome.add_argument('--headless')
with webdriver.Chrome(options=options_chrome) as browser:
    browser.get('https://parsinger.ru/scroll/2/index.html')
    input_tags = browser.find_elements(By.TAG_NAME, 'input')
    for box in input_tags:
        box.send_keys(Keys.DOWN)
        box.click()

    results = browser.find_elements(By.TAG_NAME, 'span')
    print(sum(map(lambda x: int(x.text) if x.text.isdigit() else 0, results)))

# 13310