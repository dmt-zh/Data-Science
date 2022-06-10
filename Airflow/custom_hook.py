# Импорт библиотек
from airflow.models.baseoperator import BaseOperator
from airflow.hooks.sqlite_hook import SqliteHook
from airflow.operators.python import PythonOperator
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow import DAG
from datetime import timedelta
from airflow.utils.dates import days_ago
import pandas as pd


# Определим кастомный хук
class FileSQLiteTransferHook(SqliteHook):

    # Метод считывает данные и возвращает датафрейм
    def get_pandas_df(self, url_or_path):
        data = pd.read_csv(url_or_path)
        return data

    # Метод записывает датафрейм в БД
    def insert_df_to_db(self, data):
        data.to_sql('currency', con=self.get_conn(), if_exists='replace', index=False)


# Определим кастомный оператор
class FileSQLiteTransferOperator(BaseOperator):
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.hook = None
        self.path = path

    # Метод выполняет кастомный хук
    def execute(self, context):
        self.hook = FileSQLiteTransferHook()
        data = self.hook.get_pandas_df(self.path)
        self.hook.insert_df_to_db(data)


# Определим ДАГ
dag = DAG('dag', schedule_interval=timedelta(days=1), start_date=days_ago(1))

# Запуск оператора
t1 = FileSQLiteTransferOperator(
  task_id='transfer_data',
  path='https://raw.githubusercontent.com/dm-novikov/stepik_airflow_course/main/data_new/2021-01-04.csv',
  dag=dag)