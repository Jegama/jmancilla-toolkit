from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

# load environment variables
from dotenv import load_dotenv
load_dotenv()

documents = SimpleDirectoryReader('data').load_data()

# parse documents into nodes
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# construct index from nodes
index = GPTSimpleVectorIndex(nodes)

# query index
response = index.query('Where did he go to school?', mode='embedding')

# print results
print(response.response)

index.save_to_disk('index.json')

index = GPTSimpleVectorIndex.load_from_disk('index.json')