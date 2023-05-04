from llama_index import GPTSimpleVectorIndex, download_loader, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

import pandas as pd
import json, time, openai, os, re

from dotenv import load_dotenv
load_dotenv()

ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")
loader = ReadabilityWebPageReader(wait_until="networkidle")

print('\nLoading model...')

parser = SimpleNodeParser()
cs_index = GPTSimpleVectorIndex([])
error_codes_index = GPTSimpleVectorIndex([])

urls = pd.read_csv('cs_articles.csv')['urls'].tolist()
docid_to_url = {}
tokens_used = 0

start = time.time()

def get_error_codes(document):
    new_doc = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a meticulous librarian."},
            {"role": "user", "content": f"From the following text, please extract all the error codes and their descriptions, as well as the solutions. Please make sure you return all the numbers of the error codes.\n\nTitle:{document.extra_info['title']}\n{document.text}"}
        ]
    )
    response = f"Title:{document.extra_info['title']}\n{new_doc['choices'][0]['message']['content']}"
    if not os.path.exists('temp'):
        os.makedirs('temp')
    # write response into a temp file
    with open('temp/temp.txt', 'w') as f:
        f.write(response)

    return SimpleDirectoryReader('temp').load_data()

print('\nPopulating index...')
for page in urls:
    documents = loader.load_data(url=page)
    nodes = parser.get_nodes_from_documents(documents)
    docid_to_url[nodes[0].doc_id] = page
    cs_index.insert_nodes(nodes)

    # if the documents includes any mention of error codes using regex
    if re.search(r'error code', documents[0].text, re.IGNORECASE):
        print(f'\nFound error codes in {page}')
        error_codes = get_error_codes(documents[0])
        nodes_error_codes = parser.get_nodes_from_documents(error_codes)
        docid_to_url[nodes_error_codes[0].doc_id] = page
        error_codes_index.insert_nodes(nodes_error_codes)
        os.remove('temp/temp.txt')

print(f'\nIndex populated in {(time.time() - start)/60} minutes')

total_cost = (cs_index._service_context.embed_model._total_tokens_used/1000) * 0.0004
total_cost += (error_codes_index._service_context.embed_model._total_tokens_used/1000) * 0.0004
print('\nTotal cost: $', total_cost)

cs_index.save_to_disk('cs_index.json')
error_codes_index.save_to_disk('error_codes_index.json')

with open('cs_docid_to_url.json', 'w') as f:
    json.dump(docid_to_url, f)