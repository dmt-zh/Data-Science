from airflow import DAG
from airflow.sensors.http_sensor import HttpSensor
from airflow.utils.dates import days_ago


dag = DAG('dag', schedule_interval='@daily', start_date=days_ago(1))

def response_check(response):
  number = response.json()
  if number == 5:
      return True
  else:
      return False


sensor = HttpSensor(
    task_id='http_sensor',
    http_conn_id='get_random_number',
    endpoint='/integers/?num=1&min=1&max=5&col=1&base=10&format=plain',
    response_check=response_check,
    timeout=60,
    poke_interval=10,
    dag=dag)