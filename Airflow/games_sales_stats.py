import pandas as pd
from datetime import timedelta, datetime
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.models import Variable


path = '/var/lib/airflow/airflow.git/dags/a.batalov/vgsales.csv'
year = 1994 + hash(f'd-zhigalo-18') % 23
start = datetime.today().strftime('%Y-%m-%d')


default_args = {
    'owner': 'd.zhigalo',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': start,
    'schedule_interval': '0 10 * * *'
}


@dag(default_args=default_args)
def games_sales_stats():
    @task()
    def get_data():
        data = pd.read_csv(path)
        data.rename(columns={i: i.lower() for i in data.columns}, inplace=True)
        df = data[data.year == year]
        df['year'] = df.year.astype('int64')
        for i in [i for i in df.columns if df[i].dtype not in ['int64', 'float64']]:
            df[i] = df[i].astype('str')
        return df


    @task()
    def best_sold_globally(df):
        games = df.groupby('name').agg(sales=('global_sales', 'sum')).query('sales == sales.max()').index.tolist()
        return games


    @task()
    def best_genre_europe(df):
        genres = df.groupby('genre').agg(eu_sales=('eu_sales', 'sum')).query('eu_sales == eu_sales.max()').index.tolist()
        return genres


    @task()
    def over_million_sales_na(df):
        platforms_df = df.query('na_sales > 1').platform.value_counts().reset_index().rename(columns={'index': 'platform', 'platform': 'count'})
        top = platforms_df.query('count == count.max()')
        platforms = top.platform.values.tolist()
        return platforms


    @task
    def best_publisher_japan(df):
        publisher = df.groupby('publisher').agg({'jp_sales': 'mean'}).query('jp_sales == jp_sales.max()').index.tolist()
        return publisher


    @task
    def europe_vs_japan_number(df):
        number = df.groupby('name').agg({'eu_sales': 'sum', 'jp_sales': 'sum'}).query('eu_sales > jp_sales').shape[0]
        return number


    @task
    def print_stats(*args):
        context = get_current_context()
        date = context['ds']
        print(f'''
                Current date - {date}. Games stats for the year of {year}:
                {'-' * 70}
                Best sold game worldwide: {args[0]}
                Top game genres in Europe: {args[1]}
                Top platforms in North America with over 1M sales: {args[2]}
                Top publishers in Japan with the highest median sales in: {args[3]}
                Number of Games best sold in Europe than in Japan in: {args[4]}
                ''')


    loaded_data = get_data()
    first_var = best_sold_globally(loaded_data)
    second_var = best_genre_europe(loaded_data)
    third_var = over_million_sales_na(loaded_data)
    fourth_var = best_publisher_japan(loaded_data)
    fifth_var = europe_vs_japan_number(loaded_data)
    print_stats(first_var, second_var, third_var, fourth_var, fifth_var)


get_games_sales_stats = games_sales_stats()










