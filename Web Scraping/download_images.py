# Перейдите на сайт https://parsinger.ru/img_download/index.html
# На 1 из 160 картинок написан секретный код
# Напишите код, который поможет вам скачать все картинки
# В скачанных картинках найдите вручную секретный код
# Вставьте код в поле для ответа


import requests

for number in range(1, 161):
    url = f'https://parsinger.ru/img_download/img/ready/{number}.png'
    response = requests.get(url=url)
    image_name = f'{number}.png'
    with open(image_name, 'wb') as image:
        image.write(response.content)