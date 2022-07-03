# Загружаем видео при помощи requests
# Задача:
# Перейдите на сайт 'https://parsinger.ru/video_downloads'
# Скачайте видео с сайта  при помощи requests
# Определите его размер
# Напишите размер файла в мб.


import requests
import os

def download_video():
    url = 'https://parsinger.ru/video_downloads/videoplayback.mp4'
    response = requests.get(url=url, stream=True)
    with open('videofile.mp4', 'wb') as file:
        file.write(response.content)

    file_size = os.path.getsize('videofile.mp4') // 10**6
    print(f'Download is completed. File\'s size is {file_size} MB.')

if __name__ == '__main__':
    get_video = download_video()
