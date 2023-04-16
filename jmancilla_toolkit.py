from flask import Flask, request, send_file, render_template, jsonify
from flask_cors import CORS
import qrcode, os, datetime
from io import BytesIO
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if not api_key or api_key != app.config['SECRET_KEY']:
            return jsonify({'error': 'Invalid API key'}), 403
        return f(*args, **kwargs)
    return decorated_function

app = Flask(__name__)
CORS(app)

# read key from environment variable
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')

if app.config['OPENAI_API_KEY'] is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
else:
    print("OPENAI_API_KEY loaded successfully.")

from llama_index.indices.keyword_table import GPTKeywordTableIndex

personal_index = GPTKeywordTableIndex.load_from_disk('index.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_qr', methods=['POST'])
@require_api_key
def generate_qr():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return {'error': 'No text provided'}, 400

    img = qrcode.make(text)

    # Save the QR code to a BytesIO object to serve it as an image
    img_buffer = BytesIO()
    img.save(img_buffer, 'PNG')
    img_buffer.seek(0)

    return send_file(img_buffer, mimetype='image/png')

@app.route('/generate_qr_web', methods=['POST'])
def generate_qr_web():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    return jsonify({'text': text})

@app.route('/representative', methods=['POST'])
def representative():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Log the received question
    log_question(text)

    response = personal_index.query(text)

    return jsonify({'text': response.response})

def log_question(question):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp} - {question}\n"

    # Option 1: Print the log entry to the console
    print(log_entry)

    # Option 2: Write the log entry to a file
    with open('questions.log', 'a') as f:
        f.write(log_entry)


if __name__ == '__main__':
    app.run()