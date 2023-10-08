import os, openai
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# pip install google-search-results

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Roku_cs_agent import formatter, roku_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.jgmancilla.com"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############################################################################################################

import llama_index, os
from llama_index import ServiceContext, StorageContext
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index.indices.loading import load_index_from_storage

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0
)

llm_embeddings = OpenAIEmbeddings()

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=llm_embeddings
)

llama_index.set_global_service_context(service_context)

# The other computational tasks
representative_storage_context = StorageContext.from_defaults(persist_dir="index_representative")
personal_index = load_index_from_storage(representative_storage_context)
representative_query_engine = personal_index.as_query_engine()

##############################################################################################################

class Question(BaseModel):
    question: str

@app.post('/representative')
def representative(input: Question):
    response = representative_query_engine.query(input.question)
    return response

@app.post('/query_cs')
def query(input: Question):
    response = roku_agent.run(input.question)
    return response
    
@app.post('/spotlight')
def query_spotlight(input: Question):
    response = formatter.query_cs(input.question)
    return response