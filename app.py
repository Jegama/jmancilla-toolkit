import os, qrcode, openai
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# pip install google-search-results

from functools import wraps
from io import BytesIO

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

from Roku_cs_agent import formatter, roku_agent

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
app.config['SERPAPI_API_KEY'] = os.environ.get('SERPAPI_API_KEY')

##############################################################################################################

import llama_index, os
from llama_index import ServiceContext, StorageContext
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index.indices.loading import load_index_from_storage

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0
)

llm_embeddings = OpenAIEmbeddings()

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=llm_embeddings
)

llama_index.set_global_service_context(service_context)

# The other computational tasks
representative_storage_context = StorageContext.from_defaults(persist_dir="index_representative")
personal_index = load_index_from_storage(representative_storage_context)
representative_query_engine = personal_index.as_query_engine()

##############################################################################################################

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

@app.route('/representative', methods=['POST'])
def representative():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    response = representative_query_engine.query(text)
    
    return jsonify({'text': response.response})

@app.route('/query_cs', methods=['POST'])
def query():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    else:
        response = roku_agent.run(text)
        return jsonify({'text': response})
    
@app.route('/spotlight', methods=['POST'])
def query_spotlight():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    else:
        response = formatter.query_cs(text)
        return jsonify({'text': response})

if __name__ == '__main__':
    app.run(debug=True)