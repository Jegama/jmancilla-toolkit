import requests

url = "http://127.0.0.1:5000/query_cs"
data = {
    "text": "My lightbulb is offline. What should I do?"
}
response = requests.post(url, json=data, timeout=120)

output = response.json()

print('\nResponse:', output)