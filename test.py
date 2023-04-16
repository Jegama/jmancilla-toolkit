from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from llama_index.indices.keyword_table import GPTKeywordTableIndex
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

documents = SimpleDirectoryReader('data').load_data()

# define LLM
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# build index
index = GPTKeywordTableIndex.from_documents(documents, service_context=service_context)

index.save_to_disk('index.json')