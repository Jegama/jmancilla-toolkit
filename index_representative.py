from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
import time, openai, os

# load environment variables
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

start = time.time()

print('\nLoading documents...')
documents = SimpleDirectoryReader('representative').load_data()

# parse documents into nodes
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# construct index from nodes
print('\nConstructing index...')
index = VectorStoreIndex(nodes)

print(f'\nIndex populated in {(time.time() - start)/60:.3f} minutes')

total_cost = (index._service_context.embed_model._total_tokens_used/1000) * 0.0004
print('\nTotal cost: $', total_cost)

index.storage_context.persist(persist_dir='index_representative')

# query index
query_engine = index.as_query_engine()
response = query_engine.query('Where did he go to school?')
print(response.response)