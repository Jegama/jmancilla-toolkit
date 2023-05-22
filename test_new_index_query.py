from llama_index import (
    LLMPredictor,
    ServiceContext,
    ResponseSynthesizer
)
import re
from llama_index.indices.loading import load_index_from_storage
from llama_index import StorageContext
from llama_index.indices.document_summary import DocumentSummaryIndexEmbeddingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
import nest_asyncio, time
nest_asyncio.apply()

from langchain.chat_models import ChatOpenAI
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

llm_predictor_chatgpt = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size_limit=1024)
response_synthesizer = ResponseSynthesizer.from_args(optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.5))

# personal_index = GPTVectorStoreIndex.load_from_disk('index.json', service_context=service_context)

docid_to_url = pd.read_json('cs_docid_to_url.json', typ='series').to_dict()

# rebuild storage context
print("Loading index from storage...")
cs_storage_context = StorageContext.from_defaults(persist_dir="cs_index")
cs_index = load_index_from_storage(cs_storage_context)
cs_retriever = DocumentSummaryIndexEmbeddingRetriever(
    cs_index
)
cs_query_engine = RetrieverQueryEngine(
    retriever=cs_retriever,
    response_synthesizer=response_synthesizer,
)

error_storage_context = StorageContext.from_defaults(persist_dir="error_codes_index")
error_codes_index = load_index_from_storage(error_storage_context)
error_codes_retriever = DocumentSummaryIndexEmbeddingRetriever(
    error_codes_index
)
error_codes_query_engine = RetrieverQueryEngine(
    retriever=error_codes_retriever,
    response_synthesizer=response_synthesizer,
)

class SourceFormatter:
    def formater(self, response, source_nodes):
        """Get formatted sources text."""
        texts = []
        texts.append(response)
        for source_node in source_nodes[:3]:
            title = source_node.node.text.split('\n')[0]
            doc_id = source_node.node.doc_id or "None"
            try:
                # TODO add score
                source_text = f"\nSource:\nURL: <a href=\"{docid_to_url[doc_id]}\">{title}</a>"
            except:
                source_text = f"\nSource:\nFirst line: {title} \nDocID: {doc_id}"
            texts.append(source_text)
        return "\n".join(texts)
    
    def query_cs(self, question):
        response = cs_query_engine.query(question)

        return self.formater(response.response, response.source_nodes)
    
    def query_error_codes(self, question):
        response = error_codes_query_engine.query(question)
        return self.formater(response.response, response.source_nodes)
    
    def connect_to_human(self, question):
        return "Please connect with a human agent by going to https://support.roku.com/contactus.\nThank you for using Roku Support. Have a nice day!\n\nAnother alternative is connect with my human by email at jmancilla@roku.com or scheduling a call <a href=\"https://calendly.com/jgmancilla/phonecall\">here</a>."
    
    def audio_guide(self, question):
        return "When the screen reader shortcut is enabled, you can quickly press Star * button on Roku remote four times to turn the screen reader on or off from any screen.\n\nSource <a href=\"https://support.roku.com/article/231584647\">How to enable the text-to-speech screen reader on your Roku® streaming device</a>."


print("Querying...")
formatter = SourceFormatter()

start = time.time()
response = formatter.query_cs('How do I pair my remote?')
print(response)
print(f"Query took {time.time() - start} seconds\n")