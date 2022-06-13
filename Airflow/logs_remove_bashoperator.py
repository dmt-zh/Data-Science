from airflow.decorators import task, dag
from airflow.utils.dates import days_ago
from airflow.operators.bash_operator import BashOperator
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
    @task()
    def print_python_task(task_num, **context):
        time = str(context.get('execution_date'))[:19].replace('T', ' ')
        task_id = str(context['task']).strip('<>').split()[-1]
        print(f'This is task "{task_num}" with id "{task_id}", executed {time}.')

    multitasks = [print_python_task(f'task_{i}') for i in range(4)]

    remove_logs = BashOperator(
        task_id='run_after_loop',
        bash_command='rm -r root/airflow/logs'
    )
    remove_logs


run_dag = dag()