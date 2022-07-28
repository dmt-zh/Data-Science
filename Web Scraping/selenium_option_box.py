# Открываем сайт https://parsinger.ru/selenium/7/7.html с помощью selenium;
# Получаем значения всех элементов выпадающего списка;
# Суммируем(плюсуем) все значения;
# Вставляем получившийся результат в поле на сайте;
# Нажимаем кнопку и копируем длинное число;
# Вставляем конечный результат в поле ответа.



from selenium import webdriver
from selenium.webdriver.common.by import By

with webdriver.Chrome() as browser:
    browser.get('https://parsinger.ru/selenium/7/7.html')
    option_box = browser.find_element(By.ID, 'opt')
    value = sum(map(int, option_box.text.split()))

    browser.find_element(By.ID, 'input_result').send_keys(value)
    browser.find_element(By.CLASS_NAME, 'btn').click()

    message = browser.find_element(By.ID, 'result')
    print(message.text)

