import requests

url = "http://127.0.0.1:5000/query"
data = {
    "question": "How do I fix a wifi issue?"
}
response = requests.post(url, json=data, timeout=120)
print(response.json())