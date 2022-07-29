# Откройте сайт https://parsinger.ru/scroll/4/index.html с помощью Selenium;
# На сайте есть 50 кнопок, которые визуально перекрыты блоками;
# После нажатия на кнопку в id="result" появляется уникальное для каждой кнопки число;
# Цель: написать скрипт который нажимает поочерёдно все кнопки и собирает уникальные числа;
# Все полученные числа суммировать, и вставить результат в поле для ответа.



from selenium import webdriver
from selenium.webdriver.common.by import By

options_chrome = webdriver.ChromeOptions()
options_chrome.add_argument('--headless')
with webdriver.Chrome(options=options_chrome) as browser:
    browser.get('https://parsinger.ru/scroll/4/index.html')
    taps = browser.find_elements(By.CLASS_NAME, 'btn')
    acc = 0
    for button in taps:
        browser.execute_script("return arguments[0].scrollIntoView(true);", button)
        button.click()
        acc += int(browser.find_element(By.ID, 'result').text)

print(acc)
# 4479945576993