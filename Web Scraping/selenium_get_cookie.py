# Откройте сайт https://parsinger.ru/methods/3/index.html с помощью Selenium;
# На сайте есть определённое количество секретных cookie;
# Ваша задача получить все значения и суммировать их;
# Полученный результат вставить в поле для ответа.


from selenium import webdriver

options_chrome = webdriver.ChromeOptions()
options_chrome.add_argument('--headless')
with webdriver.Chrome(options=options_chrome) as browser:
    browser.get('https://parsinger.ru/methods/3/index.html')
    cookies = browser.get_cookies()
    print(sum(map(lambda x: int(x['value'].strip()), cookies)))