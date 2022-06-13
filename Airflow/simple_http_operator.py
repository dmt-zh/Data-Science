from airflow.decorators import dag
from airflow.utils.dates import days_ago
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import timedelta

default_args = {
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
    'schedule_interval': '@daily'
}


@dag(default_args=default_args)
def dag():

    get_response = SimpleHttpOperator(
                        task_id='random_number',
                        http_conn_id='random_num',
                        method='GET',
                        endpoint='/integers/?num=1&min=1&max=5&col=1&base=2&format=plain'
    )

    get_response


run_dag = dag()