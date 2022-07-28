# Откройте сайт https://parsinger.ru/selenium/6/6.html при помощи selenium;
# Решите уравнение на странице;
# Найдите и выберите в  выпадающем списке элемент с числом, которое у вас получилось после решения уравнения;
# Нажмите на кнопку;
# Скопируйте число и вставьте в поле ответа.


from selenium import webdriver
from selenium.webdriver.common.by import By

with webdriver.Chrome() as browser:
    browser.get('https://parsinger.ru/selenium/6/6.html')
    equation = str(eval(browser.find_element(By.ID, 'text_box').text))
    browser.find_element(By.XPATH, f"//option[text()={equation}]").click()
    browser.find_element(By.CLASS_NAME, 'btn').click()
    message = browser.find_element(By.ID, 'result').text
    print(message)