# Откройте сайт https://parsinger.ru/blank/modal/2/index.html при помощи Selenium;
# На сайте есть 100 buttons;
# При нажатии на одну из кнопок в  теге <p id="result">Code</p> появится код;
# Вставьте секретный код в поле для ответа.


from selenium import webdriver
from selenium.webdriver.common.by import By

with webdriver.Chrome() as browser:
    browser.get('https://parsinger.ru/blank/modal/2/index.html')
    browser.maximize_window()
    for button in browser.find_elements(By.TAG_NAME, 'input'):
        button.click()
        alert = browser.switch_to.alert
        alert.accept()
        secret = browser.find_element(By.ID, 'result').text
        if secret:
            print(secret)
            break
            
# 321968541687435564865796413874