from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext, LangchainEmbedding
from llama_index import SimpleDirectoryReader
from langchain import HuggingFaceHub
from llama_index.node_parser import SimpleNodeParser
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

print('\nLoading model...')
repo_id = "StabilityAI/stablelm-base-alpha-3b"
# repo_id = "databricks/dolly-v2-3b"

stablelm = HuggingFaceHub(repo_id=repo_id)
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

llm_predictor = LLMPredictor(llm=stablelm)

# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# load documents
print('\nLoading documents...')
documents = SimpleDirectoryReader('library').load_data()

# parse documents into nodes
print('\nParsing documents...')
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# create index
print('\nCreating index...')
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embed_model)
index = GPTSimpleVectorIndex.from_documents(nodes, service_context=service_context)

index.save_to_disk('library_index.json')



# from langchain.embeddings import HuggingFaceEmbeddings

# model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {'device': 'cuda'}
# hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


# from langchain.embeddings import TensorflowHubEmbeddings
# url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
# tf = TensorflowHubEmbeddings(model_url=url)
