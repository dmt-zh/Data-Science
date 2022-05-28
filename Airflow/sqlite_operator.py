# Импрот библиотек
import pandas as pd
from airflow import DAG
from datetime import timedelta, datetime
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from airflow.providers.sqlite.operators.sqlite import SqliteOperator

# Определяем DAG
dag = DAG(
    dag_id='dag',
    schedule_interval='@daily',
    start_date=datetime(2022, 1, 26),
)

# Создание таблицы data в sqlite
create_table_data = SqliteOperator(
    task_id='create_table_data',
    sql="""
    CREATE TABLE if not exists data (
        currency TEXT,
        value INT,
        date DATE
    );
    """,
    dag=dag,
)

# Создание таблицы currency в sqlite
create_table_currency = SqliteOperator(
    task_id='create_table_currency',
    sql="""
    CREATE TABLE if not exists currency (
        date DATE,
        code TEXT,
        rate TEXT,
        base TEXT,
        start_date DATE,
        end_date DATE
    );
    """,
    dag=dag,
)

# Вставляем данные в БД с помощью sqlite_hook
def insert_sqlite_hook(url, table_name):
    sqlite_hook = SqliteHook()
    # Скачиваем данные
    data = pd.read_csv(url)
    # Вставляем данные
    sqlite_hook.insert_rows(table=table_name, rows=data.to_records(index=False), target_fields=list(data.columns))


# Скачиваем данные с GitHub и вставляем в БД
insert_sqlite_data = PythonOperator(
    task_id='insert_sqlite_data',
    python_callable=insert_sqlite_hook,
    op_kwargs={
        'url': 'https://raw.githubusercontent.com/dm-novikov/stepik_airflow_course/main/data_new/2021-01-01.csv',
        'table_name': 'data'
    },
    dag=dag,
)

# Скачиваем данные с exchangerate и вставляем в БД
insert_sqlite_currency = PythonOperator(
    task_id='insert_sqlite_currency',
    python_callable=insert_sqlite_hook,
    op_kwargs={
        'url': 'https://api.exchangerate.host/timeseries?start_date=2021-01-01&end_date=2021-01-01&base=EUR&format=csv&symbols=USD',
        'table_name': 'currency'
    },
    dag=dag,
)

# Создает таблицу через SQLiteOperator
create_table_join = SqliteOperator(
    task_id='create_table_join',
    sql="""
    CREATE TABLE if not exists joined_data (
        date DATE,
        code TEXT,
        rate TEXT,
        base TEXT,
        value INT
    );
    """,
    dag=dag,
)

# Объединяет данные через SQLiteOperator и загружает в БД
join_data = SqliteOperator(
    task_id='join_data',
    sql="""
        INSERT INTO joined_data (date, code, rate, base, value)
        SELECT
            l.date AS date,
            code,
            rate,
            base,
            value
        FROM currency AS l
        INNER JOIN data AS r ON l.date = r.date
        """,
    dag=dag,
)

# Определяем последовательность
[create_table_data, create_table_currency, create_table_join] >> insert_sqlite_data >> insert_sqlite_currency >> join_data