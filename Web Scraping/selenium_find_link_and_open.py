# Постановка задачи.
# Открываем сайт https://parsinger.ru/selenium/2/2.html при помощи selenium;
# Применяем метод By.PARTIAL_LINK_TEXT или By.LINK_TEXT;
# Кликаем по ссылке с текстом 16243162441624;
# Результат будет ждать вас в теге <p id="result"></p>;
# Выведите найденный результат.



from selenium import webdriver
from selenium.webdriver.common.by import By


options_chrome = webdriver.ChromeOptions()
options_chrome.add_argument('--headless')
with webdriver.Chrome(options=options_chrome) as browser:
    browser.get('https://parsinger.ru/selenium/2/2.html')
    link = browser.find_element(By.PARTIAL_LINK_TEXT, '16243162441624')
    link.click()
    message = browser.find_element(By.ID, 'result')
    print(message.text)