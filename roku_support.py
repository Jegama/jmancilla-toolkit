from llama_index import GPTSimpleVectorIndex, LangchainEmbedding, LLMPredictor, ServiceContext, PromptHelper
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline
import pandas as pd
from dotenv import load_dotenv
import re, time
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

stablelm = HuggingFacePipeline.from_model_id(model_id=repo_id, task="text-generation",  model_kwargs={"max_length":1200})
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
llm_predictor = LLMPredictor(llm=stablelm)

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 512
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)
# service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = GPTSimpleVectorIndex.load_from_disk('cs_index.json', service_context=service_context)

print("\nRunning query with optimization")
start_time = time.time()
response = index.query("How do I fix a wifi issue?",  mode="embedding", optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.5, embed_model=embed_model))
end_time = time.time()
print("\nTotal time elapsed: {:.2f}".format(end_time - start_time))
print("Answer: {}".format(response))

print(format_source_node(response))

print('\nTokens to generate response\nLLM tokens $', (int(llm_predictor.last_token_usage) * 0.02))
print('Embedding tokens $', (int(embed_model.last_token_usage) * 0.0004 ))