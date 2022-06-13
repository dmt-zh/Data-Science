from airflow.decorators import dag, task
from datetime import timedelta
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
from airflow.models import Variable


default_args = {
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'start_date': days_ago(1),
    'schedule_interval': '@daily'
}

@dag(default_args=default_args)
def dag():
    @task()
    def get_connection_info():
        host = BaseHook.get_connection("custom_conn_id").host
        login = BaseHook.get_connection("custom_conn_id").login
        password = BaseHook.get_connection("custom_conn_id").password
        values = {"host": host, "login": login, "password": password}
        return values

    @task()
    def variable_set(values):
        Variable.set(key="custom_conn_id",
                     value=values,
                     serialize_json=True)

    start = variable_set(get_connection_info())

get_and_set = dag()