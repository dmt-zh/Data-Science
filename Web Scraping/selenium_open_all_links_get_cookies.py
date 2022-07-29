# Откройте сайт https://parsinger.ru/methods/5/index.html с помощью Selenium;
# На сайте есть 42 ссылки, у каждого сайта по ссылке есть cookie с определёнными сроком жизни;
# Цель: написать скрипт, который сможет найти среди всех ссылок страницу с самым длинным сроком жизни cookie и получить с этой страницы число;
# Вставить число в поле для ответа.



from selenium import webdriver
from selenium.webdriver.common.by import By

options_chrome = webdriver.ChromeOptions()
options_chrome.add_argument('--headless')
with webdriver.Chrome(options=options_chrome) as browser:
    browser.get('https://parsinger.ru/methods/5/index.html')
    links = list(map(lambda x: x.get_attribute('href'), browser.find_elements(By.TAG_NAME, 'a')))
    code = ''
    max_expiry = 0

    for link in links:
        browser.get(link)
        expiry = int(browser.get_cookie('expiry')['expiry'])
        if expiry > max_expiry:
            code = int(browser.find_element(By.ID, 'result').text)
            max_expiry = expiry

print(code)