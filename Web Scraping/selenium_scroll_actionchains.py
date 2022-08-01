# Откройте сайт 'http://parsinger.ru/infiniti_scroll_2/' с помощью Selenium;
# На сайте есть список из 100 элементов, которые генерируются при скроллинге;
# Необходимо прокрутить окно в самый низ;
# Цель: получить все значение в элементах, сложить их;
# Получившийся результат вставить в поле ответа.



from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

unique = []
total = 0

with webdriver.Chrome() as browser:
    action = ActionChains(browser)
    browser.get('http://parsinger.ru/infiniti_scroll_2/')

    scrolling = True
    while scrolling:
        scrolling = False
        for tag in browser.find_element(By.ID, 'scroll-container').find_elements(By.TAG_NAME, 'p'):
            if tag.get_attribute('id') not in unique:
                action.scroll(650, 210, 0, 100).perform()
                total += int(tag.text)
                unique.append(tag.get_attribute('id'))
                scrolling = True

    print(total)

# 499917600