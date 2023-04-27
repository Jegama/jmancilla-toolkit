import requests

url = "http://127.0.0.1:5000/query"
data = {
    "question": "My lightbulb is offline. What should I do?"
}
response = requests.post(url, json=data, timeout=120)

output = response.json()
elapsed_time = output['elapsed_time']

print('\nResponse:', output['response'])
print('\n', output['formatted_sources'])
print(f'\nElapsed time: {elapsed_time:.2f} seconds')