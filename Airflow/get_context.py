from airflow.decorators import dag, task
from datetime import datetime

default_args = {
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'end_date': datetime(2021, 1, 10),
    'schedule_interval': '@daily'
}


@dag(default_args=default_args)
def get_context():
    @task()
    def python_task(hello, date, **context):
        print(hello)
        print(f'Execution date: {date}')
        task_id = str(context['task']).strip('<>').split()[-1]
        print(f'Task name: {task_id}')

    @task()
    def get_execution_date(**contex):
        return contex['ds']

    my_func = python_task('Hello World!', get_execution_date())

get_context_printed = get_context()


