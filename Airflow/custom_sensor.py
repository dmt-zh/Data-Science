# Импорт библиотек
from airflow import DAG
from datetime import timedelta
from airflow.utils.dates import days_ago
from airflow.sensors.base import BaseSensorOperator
import numpy as np


# Определим кастомный сенсор
class CustomSensor(BaseSensorOperator):
    def poke(self, context):
        return_value = np.random.binomial(1, 0.3)
        return bool(return_value)


# Определим ДАГ
dag = DAG('dag', schedule_interval=timedelta(days=1), start_date=days_ago(1))

# Создадим параллельные задачи с использованием CustomSensor
tasks = [CustomSensor(task_id=f'sensor_{i}', poke_interval=4, timeout=50, mode='reschedule', soft_fail=True, dag=dag) for i in range(1, 4)]