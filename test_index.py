from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext, LangchainEmbedding
from langchain import HuggingFaceHub
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

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

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embed_model)

personal_index = GPTSimpleVectorIndex.load_from_disk('index.json', service_context=service_context)
response = personal_index.query('What does he do at Roku?', response_mode="tree_summarize")

print(response.response)