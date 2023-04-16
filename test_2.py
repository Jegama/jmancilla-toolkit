from llama_index.indices.keyword_table import GPTKeywordTableIndex

from dotenv import load_dotenv
load_dotenv()

personal_index = GPTKeywordTableIndex.load_from_disk('index.json')
response = personal_index.query('What is his experience in machine learning?')

print(response.response)