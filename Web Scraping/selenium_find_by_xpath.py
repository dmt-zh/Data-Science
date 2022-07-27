# Откройте сайт https://parsinger.ru/selenium/3/3.html;
# Извлеките данные из каждого  второго тега <p>;
# Сложите все значения, их всего 100 шт;
# Выведите на экран получившийся результат.



from selenium import webdriver
from selenium.webdriver.common.by import By

options_chrome = webdriver.ChromeOptions()
options_chrome.add_argument('--headless')
with webdriver.Chrome(options=options_chrome) as browser:
    browser.get('https://parsinger.ru/selenium/3/3.html')
    tags = browser.find_elements(By.XPATH, "//div[@class='text']/p[2]")
    acc = 0
    for tag in tags:
        acc += int(tag.text)
    print(acc)
    
# 149494128600