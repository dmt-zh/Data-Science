from airflow.utils.dates import days_ago
from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
from airflow.operators.python import get_current_context
import telegram


def send_message(context):
    chat_id = *****************
    bot_tocken = '**********************************************'
    time = str(context.get('execution_date'))[:19].replace('T', ' ')
    dag_id = context.get('dag').dag_id
    task = context.get('task').task_id
    try:
        message = f"\ud83d\ude21 Task '{task}' has failed while running DAG '{dag_id}'. Execution time: {time}."
        bot = telegram.Bot(token=bot_tocken)
        bot.send_message(chat_id=chat_id, text=message)
    except:
        pass


default_args = {
    'depends_on_past': False,
    'start_date': days_ago(1),
    'schedule_interval': '@daily'
}


@dag(default_args=default_args)
def dag():
    @task(on_failure_callback=send_message)
    def raise_exception():
        raise AirflowException()

    msg = raise_exception()

get_started = dag()