from llama_index import GPTSimpleVectorIndex, download_loader
from llama_index.node_parser import SimpleNodeParser
import pandas as pd
import json, time

from dotenv import load_dotenv
load_dotenv()

ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")
loader = ReadabilityWebPageReader(wait_until="networkidle")

print('\nLoading model...')

parser = SimpleNodeParser()
index = GPTSimpleVectorIndex([])

urls = pd.read_csv('cs_articles.csv')['urls'].tolist()
docid_to_url = {}
tokens_used = 0

start = time.time()

print('\nPopulating index...')
for page in urls:
    documents = loader.load_data(url=page)
    nodes = parser.get_nodes_from_documents(documents)
    docid_to_url[nodes[0].doc_id] = page
    index.insert_nodes(nodes)

print(f'\nIndex populated in {(time.time() - start)/60} minutes')

totat_cost = (index._service_context.embed_model._total_tokens_used/1000) * 0.0004
print('\nTotal cost: $', totat_cost)

index.save_to_disk('cs_index.json')
with open('cs_docid_to_url.json', 'w') as f:
    json.dump(docid_to_url, f)