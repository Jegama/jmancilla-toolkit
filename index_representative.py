from llama_index import VectorStoreIndex, SimpleDirectoryReader
import time, openai, os

# load environment variables
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

start = time.time()

print('\nLoading documents...')
documents = SimpleDirectoryReader('representative').load_data()

# construct index from nodes
print('\nConstructing index...')
index = VectorStoreIndex(documents)

print(f'\nIndex populated in {(time.time() - start)/60:.3f} minutes')

index.storage_context.persist(persist_dir='index_representative')

# query index
query_engine = index.as_query_engine()
response = query_engine.query('Does he has experience in ML?')
print(response.response)