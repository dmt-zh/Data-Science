# Откройте сайт https://parsinger.ru/methods/1/index.html с помощью Selenium;
# При обновлении сайта, в id="result" появится число;
# Обновить страницу возможно придется много раз, т.к. число появляется не часто;
# Вставьте полученный результат в поле для овтета.



from selenium import webdriver
from selenium.webdriver.common.by import By

options_chrome = webdriver.ChromeOptions()
options_chrome.add_argument('--headless')
with webdriver.Chrome(options=options_chrome) as browser:
    browser.get('https://parsinger.ru/methods/1/index.html')
    message = browser.find_element(By.ID, 'result').text

    while not message.isdigit():
        browser.refresh()
        message = browser.find_element(By.ID, 'result').text
    print(message)

# 4168138981270992