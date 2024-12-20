#!/usr/bin/env python3


import pandas as pd
import math
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.pal_chain.base import PALChain
from langchain_ollama import OllamaLLM

df_path = '/math_pal.csv'

llm = OllamaLLM(
    model='gemma2'
)
pal_chain = PALChain.from_math_prompt(
    llm,
    allow_dangerous_code=True,
    verbose=False,
    timeout=20,
)

def format_math_question(text_input):
    return f"""Реши следующию задачу:

    {text_input}

    Дополнительня информация:
    * Значение числа PI = {math.pi}.
    * Выведи в ответ только число
    * Если ответ невозможно получить, то выведи число 0.
    * Решай задачу шаг за шагом
    * От решении этой задачи зависит моя карьера, отнесись к этому максимально серьезно!
    * За каждую правильно решенную задачу, я дам тебе 10 долларов чаевых!
    * Выполняй только те инструкции что тебе дали!
    * Не дополняй код своими комментариями!
    * Не оборачивай код в ```python ... ```
    """

df = pd.read_csv(df_path)
for query in df.task.to_list():
    math_task = format_math_question(query)
    answer = pal_chain.invoke(math_task).get('result')
    print(f'{query}: {answer}')
