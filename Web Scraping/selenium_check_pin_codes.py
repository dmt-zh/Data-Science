# Откройте сайт https://parsinger.ru/blank/modal/3/index.html при помощи Selenium;
# На сайте есть 100 buttons;
# При нажатии на любую кнопку появляется confirm с пин-кодом;
# Текстовое поле под кнопками проверяет правильность пин-кода;
# Ваша задача, найти правильный пин-код и получить секретный код;
# Вставьте секретный код в поле для ответа.



from selenium import webdriver
from selenium.webdriver.common.by import By

options_chrome = webdriver.ChromeOptions()
options_chrome.add_argument('--headless')
with webdriver.Chrome(options=options_chrome) as browser:
    browser.get('https://parsinger.ru/blank/modal/3/index.html')
    browser.maximize_window()
    input_form = browser.find_element(By.ID, 'input')
    check_button = browser.find_element(By.ID, 'check')
    for button in browser.find_element(By.CLASS_NAME, 'main').find_elements(By.CLASS_NAME, 'buttons'):
        button.click()
        confirm = browser.switch_to.alert
        pin_code = confirm.text
        confirm.accept()
        input_form.send_keys(pin_code)
        check_button.click()
        secret_code = browser.find_element(By.ID, 'result').text
        if secret_code.isdigit():
            print(secret_code)
            break

# 867413857416874163897546183542