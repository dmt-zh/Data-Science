# Откройте сайт https://parsinger.ru/html/index1_page_1.html тренажёр;
# Напишите асинхронный код, который обработает все карточки(160шт);
# Необходимо вычислить общий размер скидки для всех товаров в рублях;



from aiohttp_retry import RetryClient, ExponentialRetry
import aiohttp
import asyncio
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class AsyncItemsParser:
    def __init__(self, domain):
        self._domain = domain
        self._categories = []
        self._pages = []
        self._total_discount = 0

    @staticmethod
    def _make_soup(url):
        response = requests.get(url)
        response.encoding = 'utf-8'
        return BeautifulSoup(response.text, 'lxml')

    def _get_categories_links(self, url):
        soup = self._make_soup(url)
        make_link = lambda x: self._domain + x['href']
        for tag in soup.find('div', class_='nav_menu').find_all('a'):
                self._categories.append(make_link(tag))

    def _get_pages_links(self, categories):
        make_link = lambda x: self._domain + x['href']
        for category_link in categories:
            soup = self._make_soup(category_link)
            for pagen_link in soup.find('div', class_='pagen').find_all('a'):
                self._pages.append(make_link(pagen_link))

    async def _get_data(self, session, link):
        retry_options = ExponentialRetry(attempts=5)
        retry_client = RetryClient(retry_options=retry_options, client_session=session, start_timeout=0.5)

        async with retry_client.get(link) as page_response:
            if page_response.ok:
                response = await page_response.text()
                soup = BeautifulSoup(response, 'lxml')
                make_link = lambda x: self._domain + x['href']
                items = tuple(make_link(tag.a) for tag in soup.find_all('div', class_='sale_button'))
                for item_link in items:
                    async with session.get(item_link) as item_response:
                        response_2 = await item_response.text()
                        soup_item = BeautifulSoup(response_2, 'lxml')
                        price = soup_item.find('span', id='price').text.split()[0].strip()
                        old_price = soup_item.find('span', id='old_price').text.split()[0].strip()
                        in_stock = soup_item.find('span', id='in_stock').text.split()[-1].strip()
                        self._total_discount += (int(old_price) - int(price)) * int(in_stock)

    async def _run_async_part(self):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for link in tqdm(self._pages, desc='Обработка страниц: '):
                task = asyncio.create_task(self._get_data(session, link))
                tasks.append(task)
            await asyncio.gather(*tasks)

    def __call__(self, url, *args, **kwargs):
        self._get_categories_links(url)
        self._get_pages_links(self._categories)

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(self._run_async_part())
        print(f'Общая скидка на все товары: {self._total_discount} руб.')


parse_site = AsyncItemsParser('https://parsinger.ru/html/')
parse_site('https://parsinger.ru/html/index1_page_1.html')