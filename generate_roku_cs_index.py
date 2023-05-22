from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    ResponseSynthesizer
)
from llama_index.indices.document_summary import GPTDocumentSummaryIndex
from langchain.chat_models import ChatOpenAI

from llama_index.node_parser import SimpleNodeParser
from playwright.sync_api import sync_playwright

import json, time, openai, os, re, requests

from dotenv import load_dotenv
load_dotenv()

# if temp forlder doesn't exist, create it
if not os.path.exists('temp'):
    os.makedirs('temp')

def get_error_codes(document):
    # get first line of file temp/temp.txt
    with open('temp/temp.txt', 'r', encoding='utf-8') as f:
        title = f.readline()

    new_doc = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a meticulous librarian."},
            {"role": "user", "content": f"From the following text, please extract all the error codes and their descriptions, as well as the solutions. Please make sure you return all the numbers of the error codes.\n\nTitle:{title}\n{document.text}"}
        ]
    )
    response = f"Title:{title}\n{new_doc['choices'][0]['message']['content']}"
    # write response into a temp file
    with open('temp/temp.txt', 'w', encoding='utf-8') as f:
        f.write(response)

    return SimpleDirectoryReader('temp').load_data()

def extract_text_from_div(url, class_name):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()

        page.goto(url)
        page.wait_for_load_state('networkidle')

        div_element = page.query_selector(f".{class_name}")
        if div_element:
            text_content = div_element.inner_text()
        else:
            print(f"No element found with class: {class_name}")

        browser.close()
    with open('temp/temp.txt', 'w', encoding='utf-8') as f:
        f.write(text_content)

    return SimpleDirectoryReader('temp').load_data()

def find_urls_in_webpage(url):
    # Fetch the web page content
    response = requests.get(url)
    webpage_content = response.text

    # Define the regular expression pattern
    pattern = r'https:\/\/support\.roku\.com\/article\/\d+'

    # Find all URLs matching the pattern
    urls = re.findall(pattern, webpage_content)

    return urls

urls = find_urls_in_webpage('https://support.roku.com/sitemap.xml')

# remove duplicates from list
urls = list(dict.fromkeys(urls))
print(f'\nFound {len(urls)} unique urls')

parser = SimpleNodeParser()

# # LLM Predictor (gpt-3.5-turbo)
llm_predictor_chatgpt = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size_limit=1024)

# default mode of building the index
summary_query = (
    "Give a concise summary of this document in bullet points. Also describe some of the questions that this document can answer. "
)
response_synthesizer = ResponseSynthesizer.from_args(response_mode="tree_summarize", use_async=True)
cs_index = GPTDocumentSummaryIndex([], service_context=service_context, response_synthesizer=response_synthesizer, summary_query=summary_query)
error_codes_index = GPTDocumentSummaryIndex([], service_context=service_context, response_synthesizer=response_synthesizer)

docid_to_url = {}
tokens_used = 0

start = time.time()

print('\nPopulating index...')
for page in urls:
    documents = extract_text_from_div(page, 'article-content-wrapper')
    with open('temp/temp.txt', 'r', encoding='utf-8') as f:
        title = f.readline()
    print(f'\nProcessing {page} - {title}')
    nodes = parser.get_nodes_from_documents(documents)
    docid_to_url[nodes[0].doc_id] = page
    cs_index.insert_nodes(nodes)

    # if the documents includes any mention of error codes using regex
    if re.search(r'error code', documents[0].text, re.IGNORECASE):
        print(f'Found error codes in {page}')
        error_codes = get_error_codes(documents[0])
        nodes_error_codes = parser.get_nodes_from_documents(error_codes)
        docid_to_url[nodes_error_codes[0].doc_id] = page
        error_codes_index.insert_nodes(nodes_error_codes)
    
    os.remove('temp/temp.txt')

cs_index.storage_context.persist('index')

print(f'\nIndex populated in {(time.time() - start)/60} minutes')

total_cost = (cs_index._service_context.embed_model._total_tokens_used/1000) * 0.0004
total_cost += (error_codes_index._service_context.embed_model._total_tokens_used/1000) * 0.0004
print('\nTotal cost: $', total_cost)

cs_index.storage_context.persist(persist_dir='cs_index')
error_codes_index.storage_context.persist(persist_dir='error_codes_index')

with open('cs_docid_to_url.json', 'w') as f:
    json.dump(docid_to_url, f)