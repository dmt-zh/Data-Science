# Откройте сайт https://parsinger.ru/asyncio/aiofile/2/index.html, на нём есть 50 ссылок, в каждой ссылке лежит по 10 изображений;
# Ваша задача: Написать асинхронный код который скачает все уникальные изображения которые там есть (они повторяются, а уникальных всего 449) ;
# Вставьте размер всех скачанных изображений в поле для ответа;
# Асинхронный код должен обработать все ссылки и скачать все изображения примерно за 20-30 сек, скорость зависит от скорости вашего интернет соединения.



import aiofiles
import asyncio
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry
from bs4 import BeautifulSoup
import os
import re
import requests
from tqdm import tqdm



class AsyncImagesDowloader:
    def __init__(self):
        self._root = None
        self._total_size = 0
        self._links = []

    def _get_all_links(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        raw_links = map(lambda attr: attr['href'], soup.find_all('a'))
        for link in raw_links:
            self._links.append(self._root + link)

    async def _download_img(self, session, url, img_name):
        async with aiofiles.open(f'images/{img_name}.jpg', 'wb') as fout:
            async with session.get(url) as response:
                async for chunck in response.content.iter_chunked(3072):
                    await fout.write(chunck)

    async def _find_images(self, session, url, pbar):
        retry_options = ExponentialRetry(attempts=5)
        retry_client = RetryClient(retry_options=retry_options, client_session=session, start_timeout=0.5)

        async with retry_client.get(url) as page_response:
            if page_response.ok:
                p_response = await page_response.text()
                p_soup = BeautifulSoup(p_response, 'lxml')
                images_links = map(lambda x: x['src'], p_soup.find_all('img'))
                for img_url in images_links:
                    img_name = re.search(r'(?<=img\/)(?:.+)(?=\.jpg)', img_url).group()
                    await self._download_img(retry_client, img_url, img_name)
                    pbar.update(1)

    async def _parse_site(self):
        async with aiohttp.ClientSession() as session:
            pbar = tqdm(total=500, desc='Обработано изображений: ')
            tasks = []
            for link in self._links:
                task = asyncio.create_task(self._find_images(session, link, pbar))
                tasks.append(task)
            await asyncio.gather(*tasks)
            pbar.close()

    def _get_folder_size(self, filepath):
        for root, dirs, files in os.walk(filepath):
            for f in files:
                self._total_size += os.path.getsize(os.path.join(root, f))

    def __call__(self, default_args):
        self._root = default_args.get('domain')
        self._get_all_links(default_args.get('url'))
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(self._parse_site())
        self._get_folder_size(default_args.get('folder_path'))
        print(f'Общий размер скачанных изображений: {self._total_size}')


parser_args = {
    'domain': 'https://parsinger.ru/asyncio/aiofile/2/',
    'url': 'https://parsinger.ru/asyncio/aiofile/2/index.html',
    'folder_path': 'images'
}

image_dowloader = AsyncImagesDowloader()
image_dowloader(default_args=parser_args)