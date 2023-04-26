from llama_index import GPTSimpleVectorIndex, download_loader, LangchainEmbedding, LLMPredictor, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.node_parser import SimpleNodeParser
from llama_index.utils import truncate_text
from langchain import HuggingFacePipeline
import pandas as pd
from dotenv import load_dotenv
import re
load_dotenv()

docid_to_url = pd.read_json('cs_docid_to_url.json', typ='series').to_dict()

def format_source_node(response_):
        """Get formatted sources text."""
        texts = []
        for source_node in response_.source_nodes:
            title = re.search(r'title:\s*(.*?)\s*\|', source_node.node.get_text()).group(1)
            doc_id = source_node.node.doc_id or "None"
            source_text = f"\nSource:\nTitle: {title}\nConfidence: {source_node.score:.3f}\nURL: {docid_to_url[doc_id]}"
            texts.append(source_text)
        return "\n\n".join(texts)

print('\nLoading model...')
repo_id = "stabilityai/stablelm-tuned-alpha-7b"
# repo_id = "databricks/dolly-v2-3b"

stablelm = HuggingFacePipeline.from_model_id(model_id=repo_id, task="text-generation")
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
llm_predictor = LLMPredictor(llm=stablelm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

index = GPTSimpleVectorIndex.load_from_disk('cs_index.json', service_context=service_context)

response = index.query('How do I fix a wifi issue?', response_mode="tree_summarize")

print(response)

print(format_source_node(response))

