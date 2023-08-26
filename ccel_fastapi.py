import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class Question(BaseModel):
    question: str

from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext

ccel_storage_context = StorageContext.from_defaults(persist_dir='ccel_index')
ccel_index = load_index_from_storage(ccel_storage_context)
ccel_query_engine = ccel_index.as_query_engine(
    response_mode='refine',
    verbose=True,
    similarity_top_n=5
)

@app.post("/query")
def query(input: Question):
    response = ccel_query_engine.query(input.question)
    return response.response