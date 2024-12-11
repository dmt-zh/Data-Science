#!/usr/bin/env python3

import pandas as pd
import regex
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from typing import Sequence

####################################################################################################################

def preprocess(sent: str) -> str:
    sent = BeautifulSoup(sent, 'html.parser').get_text()
    sent = regex.sub(r'\[.+?\]', '', sent)
    sent = regex.sub(r'({{)(.+?)(?<=[\|])([^|]+?)(}})', r'\3', sent)
    sent = regex.sub(r'\s{2,}', '', sent)
    return sent.strip()

####################################################################################################################

def format_docs(docs: Sequence[Document]) -> str:
    return '\n\n'.join([doc.page_content for doc in docs])

####################################################################################################################

url = 'https://ru.wikipedia.org/wiki/Ганнибал,_Абрам_Петрович'
response = requests.get(url)
response.encoding= 'utf8'
soup = BeautifulSoup(response.text, 'lxml')
page_text = [preprocess(tag.text) for tag in soup.find_all('p')]
documents = [Document(page_content=doc, metadata={}) for doc in page_text]

embedder = HuggingFaceBgeEmbeddings(
    model_name='intfloat/multilingual-e5-large',
    encode_kwargs={'normalize_embeddings': True},
    embed_instruction='query: ',
)
vectorstore = FAISS.from_documents(
    documents,
    embedder
)
retriever = vectorstore.as_retriever(
    search_type='similarity',
    k=5,
    score_threshold=None,
)

####################################################################################################################

template = """
###INSTRUCTIONS###

You MUST follow the instructions for answering:

- ALWAYS answer in the language of my Question.
- ALWAYS read Question carefully before answering.
- You ALWAYS will be PENALIZED for wrong and low-effort answers. 
- ALWAYS follow "Answering rules."

###Answering Rules###

Follow in the strict order:

1. USE the language of my message.
2. To answer Question use information from Context
2. I'm going to tip $1,000,000 for the best reply.
3. Your answer is critical for my career.
4. Answer the question in a natural, human-like manner.
5. Your answer MUST be less than 70 characters.

{context}

Question: {question}
"""

####################################################################################################################

prompt_template = ChatPromptTemplate.from_template(template)
gemma2_llm = OllamaLLM(model='gemma2')

qa_df_path = 'gannibal.csv'
df = pd.read_csv(qa_df_path)
answers = []
for query in df.question.to_list():
    top_docs = retriever.get_relevant_documents(query.strip())
    context = format_docs(top_docs)
    message = prompt_template.format_messages(question=query.strip(), context=context)
    answer = gemma2_llm.invoke(message).strip()
    answers.append(answer)

df['answer'] = answers
df[['question', 'answer']].to_csv(qa_df_path, index=False)
