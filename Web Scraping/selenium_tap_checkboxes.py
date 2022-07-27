# Откройте сайт https://parsinger.ru/selenium/4/4.html;
# Установите все чек боксы в положение checked при помощи selenium и метода click();
# Когда все чек боксы станут активны, нажмите на кнопку;
# Скопируйте число которое появится на странице;
# Результат появится в <p id="result">Result</p>;
# Вставьте число в поле для ответа.


from selenium import webdriver
from selenium.webdriver.common.by import By

with webdriver.Chrome() as browser:
    browser.get('https://parsinger.ru/selenium/4/4.html')
    check_boxes = browser.find_elements(By.CLASS_NAME, 'check')
    for cb in check_boxes:
        cb.click()
    button = browser.find_element(By.CLASS_NAME, 'btn').click()
    message = browser.find_element(By.ID, 'result')
    print(message.text)

# 3,1415926535897932384626433832795028841971