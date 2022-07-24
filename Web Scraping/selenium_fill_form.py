# Суть задачи проста( у вас будет всего 5 секунд для того чтобы получить результат, поэтому подумайте над кодом)

# Открыть сайт https://parsinger.ru/selenium/1/1.html с помощью selenium;
# Заполнить все существующие поля;
# Нажмите на кнопку;
# Скопируйте результат который появится рядом с кнопкой в случае если вы уложились в 5 секунд;
# Вставьте результат в поле ниже.



import random
import time
from faker import Faker
from selenium import webdriver
from selenium.webdriver.common.by import By


fkr = Faker(['ru_Ru'])
info = [fkr.first_name_male(), fkr.middle_name_male(), fkr.last_name_male(), random.randint(18, 60), fkr.city(), fkr.email()]

with webdriver.Chrome() as browser:
    browser.get('https://stepik-parsing.ru/selenium/1/1.html')
    input_form = browser.find_elements(By.CLASS_NAME, 'form')
    for k, v in zip(input_form, info):
        k.send_keys(v)
    time.sleep(2)
    browser.find_element(By.ID, 'btn').click()
    time.sleep(1)
    secret_code = browser.find_element(By.ID, 'result').text
    print(secret_code)