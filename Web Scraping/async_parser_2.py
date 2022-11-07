# 1. Откройте сайт https://parsinger.ru/asyncio/create_soup/1/index.html, там есть 500 ссылок,
#    секретный код лежит только на четырёх из них;
# 2. Напишите асинхронный код, который найдёт все четыре кода и суммирует их;
# 3. Суммируйте все полученный цифры и вставьте результат в поле для ответа.



import aiohttp
import asyncio
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class AsyncNumbersFinder:
    def __init__(self, root):
        self._root = root
        self._links = []
        self.total_sum = 0

    def _get_all_links(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        raw_links = map(lambda attr: attr['href'], soup.find_all('a'))
        for link in raw_links:
            self._links.append(self._root + link)

    async def _find_number(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.ok:
                    soup = BeautifulSoup(await response.text(), 'lxml')
                    self.total_sum += int(soup.find('p').text)

    def __call__(self, url, *args, **kwargs):
        self._get_all_links(url)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        tasks = tqdm(map(self._find_number, self._links), total=len(self._links), desc='Поиск чисел: ')
        asyncio.run(asyncio.wait(tasks))


links_parser = AsyncNumbersFinder(root='https://parsinger.ru/asyncio/create_soup/1/')
links_parser(url='https://parsinger.ru/asyncio/create_soup/1/index.html')
print(links_parser.total_sum)