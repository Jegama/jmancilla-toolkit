import requests, json

url = "http://127.0.0.1:5000/representative"

while True:
    text_input = input("User: ")
    data = {
        "text": text_input
    }
    response = requests.post(url, json=data, timeout=120, stream=True)
    for line in response.iter_lines():
        if line:  # filter out keep-alive new lines
            output = json.loads(line.decode()).get('text')
            print(f'Agent: {output}')
