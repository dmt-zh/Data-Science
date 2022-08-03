# Откройте сайт https://parsinger.ru/blank/modal/4/index.html при помощи Selenium;
# На сайте есть список пин-кодов и только один правильный;
# Для проверки пин-кода используйте кнопку "Проверить"
# Ваша задача, найти правильный пин-код и получить секретный код;
# Вставьте секретный код в поле для ответа.



from selenium import webdriver
from selenium.webdriver.common.by import By

options_chrome = webdriver.ChromeOptions()
options_chrome.add_argument('--headless')
with webdriver.Chrome(options=options_chrome) as browser:
    browser.get('https://parsinger.ru/blank/modal/4/index.html')
    check_button = browser.find_element(By.ID, 'check')
    result = browser.find_element(By.ID, 'result')
    pin_codes = map(lambda x: x.text, browser.find_elements(By.CLASS_NAME, 'pin'))
    for pin in pin_codes:
        check_button.click()
        confirm = browser.switch_to.alert
        confirm.send_keys(pin)
        confirm.accept()
        if result.text.isdigit():
            print(result.text)
            break