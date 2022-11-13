# Откройте сайт https://parsinger.ru/asyncio/aiofile/3/index.html, на нём есть 100 ссылок, в каждой из них есть
# ещё 10 ссылок, в каждой из 10 ссылок есть 8-10 изображений, структура как на картинке ниже;
# Ваша задача: Написать асинхронный код, который скачает все уникальные изображения, которые там есть (они повторяются,
# в это задании вам придётся скачать 2615 изображений);
# Вставьте размер всех скачанных изображений в поле для ответа;


import aiofiles
import asyncio
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry
from bs4 import BeautifulSoup
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging


class AsyncDeepImagesDowloader:
    def __init__(self, root):
        self.root = root
        self.total_size = 0
        self.level2_links = []
        self.images_links = set()

    def get_level1_links(self, url):
        soup = BeautifulSoup(requests.get(url).text, 'lxml')
        return map(lambda tag: self.root + tag['href'], soup.find_all('a'))

    def get_level2_links(self, url):
        soup = BeautifulSoup(requests.get(url).text, 'lxml')
        image_pages = [''.join((self.root, 'depth2/', x['href'])) for x in soup.find_all('a')]
        self.level2_links.extend(image_pages)

    async def download(self, session, url, pbar):
        async with aiofiles.open(f'images/{url.split("/")[6]}.jpg', 'wb') as fout:
            async with session.get(url) as response:
                pbar.update()
                async for chunck in response.content.iter_chunked(3072):
                    await fout.write(chunck)

    async def get_images(self, session, url, pbar):
        retry_options = ExponentialRetry(attempts=5)
        retry_client = RetryClient(retry_options=retry_options, client_session=session, start_timeout=0.5)

        async with retry_client.get(url) as page_response:
            if page_response.ok:
                p_response = await page_response.text()
                p_soup = BeautifulSoup(p_response, 'lxml')
                links = list(map(lambda x: x['src'], p_soup.find_all('img')))
                pbar.update()
                self.images_links.update(links)

    async def parse_site(self):
        async with aiohttp.ClientSession() as session:
            pbar = tqdm(total=len(self.level2_links), desc='Обработка ссылок 2 уровня : ', colour='WHITE')
            links_tasks = []
            for link in self.level2_links:
                link_task = asyncio.create_task(self.get_images(session, link, pbar))
                links_tasks.append(link_task)
            await asyncio.gather(*links_tasks)
            pbar.close()

            pbar = tqdm(total=len(self.images_links), desc='Скачивание изображений: ', colour='YELLOW')
            download_tasks = []
            for img in self.images_links:
                img_task = asyncio.create_task(self.download(session, img, pbar))
                download_tasks.append(img_task)
            await asyncio.gather(*download_tasks)
            pbar.close()

    def get_folder_size(self, filepath):
        for root, dirs, files in os.walk(filepath):
            for img in files:
                self.total_size += os.path.getsize(os.path.join(root, img))

    def __call__(self, main_page, *args, **kwargs):
        level1_links = self.get_level1_links(main_page)

        with ThreadPoolExecutor(30) as pool:
            pool.map(self.get_level2_links, tqdm(level1_links, desc='Обработка ссылок 1 уровня: ', total=100))

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(self.parse_site())

        self.get_folder_size('images')
        print(f'Общий размер скачанных изображений: {self.total_size}')



if __name__ == '__main__':
    try:
        image_dowloader = AsyncDeepImagesDowloader(root='https://parsinger.ru/asyncio/aiofile/3/')
        image_dowloader(main_page='https://parsinger.ru/asyncio/aiofile/3/index.html')
    except Exception as error:
        logging.error(f'Парсер аварийно завершил работу: {error}')