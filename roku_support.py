from llama_index import GPTSimpleVectorIndex, download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index.utils import truncate_text
from dotenv import load_dotenv
import re
load_dotenv()

def format_source_node(response_):
        """Get formatted sources text."""
        texts = []
        for source_node in response_.source_nodes:
            title = re.search(r'title:\s*(.*?)\s*\|', source_node.node.get_text()).group(1)
            doc_id = source_node.node.doc_id or "None"
            source_text = f"\nSource: Doc id: {doc_id} (Confidence: {source_node.score:.2f})\nTitle: {title}"
            texts.append(source_text)
        return "\n\n".join(texts)

# $env:PWDEBUG=0

ReadabilityWebPageReader = download_loader("ReadabilityWebPageReader")

loader = ReadabilityWebPageReader(wait_until="networkidle")

urls = ['https://support.roku.com/article/115015760328', 'https://support.roku.com/article/360011612733', 'https://support.roku.com/article/208755978']

parser = SimpleNodeParser()

index = GPTSimpleVectorIndex([])

for page in urls:
    documents = loader.load_data(url=page)
    nodes = parser.get_nodes_from_documents(documents)
    index.insert_nodes(nodes)

response = index.query('How do I fix a wifi issue?', response_mode="tree_summarize")

print(response)

print(format_source_node(response))

