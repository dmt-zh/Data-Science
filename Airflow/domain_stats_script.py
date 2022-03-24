# Импорт библиотек
import requests
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# Ссылка на данные
alexa_static = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'

# Файл с данными
domains_file = 'top-1m.csv'


# Функция считывает данные и записывает в файл
def get_data():
    raw_data = requests.get(alexa_static)
    zipfile = ZipFile(BytesIO(raw_data.content))
    data = zipfile.read(domains_file).decode('utf-8')

    with open(domains_file, 'w') as f:
        f.write(data)


# Функция возвращает 10 наиболее встречаемых доменных зон в порядке убывания
def top_domain_zones(ds):
    date = ds
    df = pd.read_csv(domains_file, names=['rank', 'domain'])
    df['d_zone'] = df.domain.str.split('.').str[-1]
    print(f'Top domain zones for date {date} :')
    result = [f'{k} - {v}' for k, v in (zip(df.d_zone.value_counts()[:10].index, df.d_zone.value_counts()[:10].values))]
    print(result)


# Функция возвращает самое длинное из имен домена в данных в алфавитном порядке
def get_longest_name(ds):
    date = ds
    df = pd.read_csv(domains_file, names=['rank', 'domain'])
    df['name_length'] = df.domain.str.split('.').str[0].str.len()
    name = df[df.name_length == df.name_length.max()].sort_values('domain').domain.values[0].split('.')[0]
    message = f'The longest name for date {date} : {name}'
    print(message)


# Функция возвращает ранг указываемого домена
def get_rank(ds):
    df = pd.read_csv(domains_file, names=['rank', 'domain'])
    date = ds
    try:
        rank = df[df.domain == 'airflow.com']['rank'].values[0]
        message = f'airflow.com rank for date {date}: {rank}'
    except:
        message = f'There\'s no airflow.com in the data today!'
    print(message)


# Дефолтные аргументы
default_args = {
    'owner': 'd.zhigalo',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2022, 3, 23)
}

# Инервал выполнения
schedule_interval = '0 8 * * *'

# Определяем DAG
dag = DAG('domain_stats_script', default_args=default_args, schedule_interval=schedule_interval)

# Определяем задачи
t1 = PythonOperator(task_id='get_data',
                    python_callable=get_data,
                    dag=dag)

t2 = PythonOperator(task_id='top_domain_zones',
                    python_callable=top_domain_zones,
                    dag=dag)

t3 = PythonOperator(task_id='get_longest_name',
                    python_callable=get_longest_name,
                    dag=dag)

t4 = PythonOperator(task_id='get_rank',
                    python_callable=get_rank,
                    dag=dag)

# Определяем последовательность
t1 >> [t2, t3, t4]
