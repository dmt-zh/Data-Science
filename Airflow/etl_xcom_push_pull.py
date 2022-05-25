# Импорт библиотек
from airflow.decorators import dag, task
from datetime import datetime
import pandas as pd
import sqlite3

# Инициализация БД
CONN = sqlite3.connect('currency.db')

# Функция записывает данные в БД
def insert_to_db(tmp_file, table_name):
    df = pd.read_csv(tmp_file)
    df.to_sql(table_name, CONN, if_exists='replace', index=False)

# Функция выполняет sql запрос
def sql_query(sql):
    query = sql.strip().lower()
    cur = CONN.cursor()
    cur.execute(sql)
    if query.startswith('select'):
        df = pd.read_sql(query, CONN)
        df.to_csv('/tmp/joined_data.csv')

# Дефолтные аргументы DAG-a
default_args = {
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'end_date': datetime(2021, 1, 4)
}

# Определяем DAG
@dag(default_args=default_args)
def dag_xcom_etl():
    # Считываем данные по обменному курсу и передаем в xcom
    @task()
    def load_currency(**context):
        date = context['ds']
        url = f"https://api.exchangerate.host/timeseries?start_date={date}&end_date={date}&base=EUR&symbols=USD&format=csv"
        data = pd.read_csv(url)
        df = data.drop(['start_date', 'end_date'], axis=1)
        df['rate'] = df['rate'].str.replace(',', '.')
        rate = df['rate'].astype('float').values.tolist()[0]
        context['ti'].xcom_push(key='exchange_rate', value=rate)

    # Считываем данные с GitHub
    @task()
    def load_data(**context):
        date = context['ds']
        url = f'https://raw.githubusercontent.com/dm-novikov/stepik_airflow_course/main/data_new/{date}.csv'
        data = pd.read_csv(url)
        data.to_csv('/tmp/data.csv')
        insert_to_db('/tmp/data.csv', 'data')

    # Объединяем данные и записываем в БД
    @task()
    def merge_data(**context):
        rate = context['ti'].xcom_pull(key='extract', task_ids='exchange_rate')
        query = "SELECT * from data"
        sql_query(query)
        data = pd.read_csv('/tmp/joined_data.csv')
        data['rate'] = rate
        data['code'] = 'USD'
        data.to_csv('/tmp/merged_data.csv')
        insert_to_db('/tmp/merged_data.csv', 'merged_data')

    load_rate = load_currency()
    load_data = load_data()
    join_data = merge_data()

# Инициализируем запуск DAG-a
get_currency = dag_xcom_etl()
