from llama_index import GPTSimpleVectorIndex, ServiceContext, LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

print('\nLoading model...')
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

service_context = ServiceContext.from_defaults(embed_model=embed_model)

print('\nLoading index...')
personal_index = GPTSimpleVectorIndex.load_from_disk('library_index.json', service_context=service_context)

response = personal_index.query('What is the second commandment?', response_mode="tree_summarize")

print(response)

print('\nSource:', response.source_nodes)