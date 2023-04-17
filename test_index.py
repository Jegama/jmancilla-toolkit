from llama_index import GPTSimpleVectorIndex

from dotenv import load_dotenv
load_dotenv()

personal_index = GPTSimpleVectorIndex.load_from_disk('index.json')
response = personal_index.query('What does he do at Roku?', response_mode="tree_summarize")

print(response.response)