# Импорт необходимых библиотек
from airflow.decorators import dag, task
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta
from airflow.utils.dates import days_ago
import pandas as pd
import sqlite3


CONN = sqlite3.connect('currency.db')

def insert_to_db(tmp_file, table_name):
    df = pd.read_csv(tmp_file)
    df.to_sql(table_name, CONN, if_exists='replace', index=False)

def sql_query(sql):
    query = sql.strip().lower()
    cur = CONN.cursor()
    cur.execute(sql)
    if query.startswith('select'):
        df = pd.read_sql(query, CONN)
        df.to_csv('/tmp/joined_data.csv')


default_args = {
    'depends_on_past': False,
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
    'schedule_interval': '@daily'
}

@dag(default_args=default_args)
def simple_elt():
    @task()
    def read_currency(date, tmp_file, table_name):
        url = f"https://api.exchangerate.host/timeseries?start_date={date}&end_date={date}&base=EUR&symbols=USD&format=csv"
        data = pd.read_csv(url)
        df = data.drop(['start_date', 'end_date'], axis=1)
        df.to_csv(tmp_file)
        insert_to_db(tmp_file, table_name)

    @task()
    def read_data(date, tmp_file, table_name):
        url = f'https://raw.githubusercontent.com/dm-novikov/stepik_airflow_course/main/data_new/{date}.csv'
        data = pd.read_csv(url)
        df = data.drop(['currency'], axis=1)
        df.to_csv(tmp_file)
        insert_to_db(tmp_file, table_name)


    currency = read_currency('2021-01-01', '/tmp/currency.csv', 'currency')
    data = read_data('2021-01-01', '/tmp/data.csv', 'data')

    join_data = PythonOperator(
        task_id='join_data',
        python_callable=sql_query,
        op_kwargs={'sql': "SELECT currency.date, code, rate, base, value from currency INNER JOIN data ON currency.date = data.date"})
    [currency, data] >> join_data

simple_elt = simple_elt()