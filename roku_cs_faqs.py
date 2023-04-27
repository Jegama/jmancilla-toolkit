from flask import Flask, request, jsonify
from llama_index import GPTSimpleVectorIndex, LangchainEmbedding, ServiceContext
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pandas as pd
from dotenv import load_dotenv
import re, time
load_dotenv()

app = Flask(__name__)


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

embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = GPTSimpleVectorIndex.load_from_disk('cs_index.json', service_context=service_context)

def get_query_result(question):
    start_time = time.time()
    response = index.query(question, optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.5, embed_model=embed_model))
    end_time = time.time()
    return {
        "response": str(response),
        "formatted_sources": format_source_node(response),
        "elapsed_time": end_time - start_time,
    }

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json(force=True)
    question = data.get('question', None)

    if not question:
        return jsonify({"error": "No question provided"}), 400

    result = get_query_result(question)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
