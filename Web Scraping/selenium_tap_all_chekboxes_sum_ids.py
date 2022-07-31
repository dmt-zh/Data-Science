# Откройте сайт https://parsinger.ru/scroll/3/ с помощью Selenium, на целевом сайте 500 тегов;
# Ваша задача, получить числовое значение  id="число" с каждого тега <input> который при нажатии вернул число;
# Суммируйте все значения и отправьте результат в поле ниже.



from selenium.webdriver import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By

options_chrome = webdriver.ChromeOptions()
options_chrome.add_argument('--headless')
with webdriver.Chrome(options=options_chrome) as browser:
    browser.get('https://parsinger.ru/scroll/3/')
    result = 0
    for div in browser.find_elements(By.CLASS_NAME, 'item'):
        btn = div.find_element(By.TAG_NAME, 'input')
        btn.send_keys(Keys.DOWN)
        btn.click()
        if div.find_element(By.TAG_NAME, 'span').text:
            result += int(btn.get_attribute('id'))
    print(result)


# 9906