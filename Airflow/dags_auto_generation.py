from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import timedelta
from airflow.utils.dates import days_ago


def create_dag(dag_id, def_args, schedule='@daily'):
    dag = DAG(dag_id, schedule_interval=schedule, default_args=def_args)
    tasks = [DummyOperator(task_id=f'task_{i}', dag=dag) for i in range(10)]
    return dag


for number in range(1, 6):
    dag_id = f'dag_{number}'

    def_args = {
        'depends_on_past': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'start_date': days_ago(1)
    }

    globals()[dag_id] = create_dag(dag_id, def_args)
