#!/usr/bin/env python3

import pandas as pd
import regex

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


qa_df_path = 'pushkin_questions.csv'
pdf_path = 'The_Daughter_of_The_Commandant.pdf'

remove_trash = regex.compile(r'(?ms)^[\w\s\d.]{,20}\s*$')
remove_empty = regex.compile(r'(?m)\n\n')

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])


loader = PyPDFLoader(pdf_path)
documents = loader.load()

for doc in documents:
    doc.page_content = remove_trash.sub('', doc.page_content).strip()
    doc.page_content = remove_empty.sub('', doc.page_content).strip()

text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n', '.'],
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)
texts = text_splitter.split_documents(documents)

embedder = HuggingFaceBgeEmbeddings(
    model_name='intfloat/multilingual-e5-large',
    encode_kwargs={'normalize_embeddings': True},
    embed_instruction='query: ',
)

vectorstore = FAISS.from_documents(texts, embedder)
retriever = vectorstore.as_retriever(
    search_type='similarity',
    k=3,
    score_threshold=None,
)

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

prompt_template = ChatPromptTemplate.from_template(template)
gemma2_llm = OllamaLLM(model='gemma2')

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
