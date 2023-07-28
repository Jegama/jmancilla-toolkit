from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext
import pandas as pd
import openai, os

# load environment variables
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

ccel_storage_context = StorageContext.from_defaults(persist_dir='ccel_index')
ccel_index = load_index_from_storage(ccel_storage_context)
ccel_query_engine = ccel_index.as_query_engine()

def get_formatted_sources_ccel(response):
    """Get formatted sources text."""
    texts = []
    books = []
    for source_node in response.source_nodes:
        if source_node.node.metadata['title'] not in books:
            books.append(source_node.node.metadata['title'])
            source_text = f"\nTitle: {source_node.node.metadata['title']}\nCreators: {source_node.node.metadata['creators']}\nScore: {source_node.score:.4f}"
            texts.append(source_text)
    return "\n\n".join(texts)


response = ccel_query_engine.query('What is the cheif end of man?')
print(response)
print(get_formatted_sources_ccel(response))