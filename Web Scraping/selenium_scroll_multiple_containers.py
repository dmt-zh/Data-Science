# Откройте сайт https://parsinger.ru/infiniti_scroll_3/ с помощью Selenium
# На сайте есть 5 окошек с подгружаемыми элементами, в каждом по 100 элементов;
# Необходимо прокрутить все окна в самый низ;
# Цель: получить все значение в каждом из окошек и сложить их;
# Получившийся результат вставить в поле ответа.


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin


values = []
with webdriver.Chrome() as browser:
    browser.get('https://parsinger.ru/infiniti_scroll_3/')
    for x in browser.find_elements(By.XPATH, '//div[contains(@class, "scroll-container")]'):
        while not x.find_elements(By.CLASS_NAME, 'last-of-list'):
            ActionChains(browser).scroll_from_origin(ScrollOrigin.from_viewport(int(x.rect['x'])+3, 550), 0, 550).perform()
        [values.append(int(y.text)) for y in x.find_elements(By.TAG_NAME, 'span')]

print(sum(values)) # 159858750