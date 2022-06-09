from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator
from random import randint

default_args = {
    'owner':'airflow',
    'start_date': datetime(2022, 2, 16)
}

dag = DAG('dag', schedule_interval='@daily', default_args=default_args)


def rand(**kwargs):
  kwargs['ti'].xcom_push(key='rand', value=randint(0, 10))


def branch(**kwargs):
    xcom_value = int(kwargs['ti'].xcom_pull(key='rand', task_ids='random_number'))
    if xcom_value > 5:
        return 'higher'
    else:
        return 'lower'


lower = DummyOperator(
    task_id = 'lower',
    dag=dag
)

higher = DummyOperator(
    task_id = 'higher',
    dag=dag
)

branch_op = BranchPythonOperator(
    task_id = 'branch_task',
    provide_context = True,
    python_callable=branch,
    dag=dag
)

random_number = PythonOperator(
    task_id = 'random_number',
    python_callable=rand,
    dag=dag
)


random_number >> branch_op >> [lower, higher]