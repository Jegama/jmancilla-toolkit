from flask import Flask, request, send_file, render_template, jsonify
from flask_cors import CORS
import qrcode, os, datetime, re, time, random
from io import BytesIO
from functools import wraps
from dotenv import load_dotenv
load_dotenv()

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

if app.config['OPENAI_API_KEY'] is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
else:
    print("OPENAI_API_KEY loaded successfully.")

##############################################################################################################

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import OpenAI, LLMChain
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import List, Union
# from elevenlabs import generate, play
from llama_index import GPTSimpleVectorIndex, LLMPredictor
from langchain.utilities import SerpAPIWrapper
import pandas as pd

personal_index = GPTSimpleVectorIndex.load_from_disk('index.json')

cs_index = GPTSimpleVectorIndex.load_from_disk('cs_index.json')

class SourceFormatter:
    def query(self, question):
        response = cs_index.query(question)
        """Get formatted sources text."""
        texts = []
        texts.append(response.response)
        for source_node in response.source_nodes:
            title = re.search(r'title:\s*(.*?)\s*\|', source_node.node.get_text()).group(1)
            doc_id = source_node.node.doc_id or "None"
            source_text = f"\nSource:\nConfidence: {source_node.score:.3f}\nTitle: {title}\nURL: {docid_to_url[doc_id]}"
            texts.append(source_text)
        return "\n".join(texts)


template = """You are a friendly Roku customer support agent. People who talk with you might not be tech-savvy; you can break down the instructions into smaller, more manageable steps. For example, instead of providing a long list of actions to take, you could break down each step and explain it thoroughly in short, simple sentences. Always refer to the context of the conversation when the user follows up with another question. If the first search doesn't work, try different keywords; for example, if the user wants to change the pin, you can search for "update pin" or "reset pin." Remember, not all problems can be solved on the device; if the article mentions "my.roku.com," they need to go to that website to change something on their account. Convert all the URLs on your response in the following format `<a href="https://support.roku.com/">Roku Support Site</a>.` If the question is unrelated to Roku, please reply politely, saying that you can only answer questions related to Roku, skip the action, and give a "Final Answer." In the same way, if a user thanks you or shows kindness, skip the action and respond accordingly. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take should be one of [{tool_names}], but prioritize the "CS Vector Index."
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 3 times, if you don't find the answer by then, politely tell the user to connect with a human agent)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. Include the source from the Observation.

Begin! Remember to be friendly and explain things thoroughly with simple language. Always make sure the source URL is correct. If you use "search," return the URL from the website where you took the answer. 

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))

docid_to_url = pd.read_json('cs_docid_to_url.json', typ='series').to_dict()

from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
)
formatter = SourceFormatter()
cs_index_config = Tool(
    func=formatter.query, 
    name=f"CS Vector Index",
    description=f"useful for when you want to answer queries that require Roku Customer Support site"
)

search = SerpAPIWrapper()
search_config = Tool(
        name = "search",
        func=search.run,
        description="useful for when you can not find the answer on the official Roku Customer Support site. You should ask targeted questions"
    )

tools = [cs_index_config, search_config]

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

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

too_short = [
    'I am sorry, I need more context to answer your question. Can you expand on your question?', 
    'Apologies, but I require additional context to provide an accurate response. Could you please elaborate on your question?', 
    "I'm sorry, I could use more information to help you better. Would you mind providing more details about your question?", 
    "My apologies, but I need a bit more background to give you a proper answer. Can you please provide more information about your query?", 
    "Pardon me, but I need more details to effectively answer your question. Could you kindly expand on your inquiry?", 
    "I'm sorry, I'd like to help but I need more context. Can you please share more specifics about your question?", 
    "Excuse me, but could you please provide more information on your question? I need some additional context to answer accurately.", 
    "My apologies, but I need a little more insight to respond to your question effectively. Would you mind elaborating on your query?", 
    "I'm sorry, but I need more clarification to give you the best possible answer. Can you please provide more details about your question?", 
    "I'm sorry, but I need more information to answer your question. Can you please provide more details about your question?", 
    "Forgive me, but I need more context to offer a helpful response. Can you please expand on your question a bit more?", 
    "I apologize, but I need further information to address your question accurately. Could you kindly give me more context?", 
    "Sorry, but to assist you better, I need more background on your question. Can you please provide more details?", 
    "I apologize, but I require more specifics to answer your question correctly. Would you mind sharing more context?", 
    "My apologies, but to give you an appropriate response, I need more information. Can you please elaborate on your question?", 
    "I'm sorry, but I need a clearer understanding of your question to provide a helpful answer. Can you please give me more context?", 
    "Excuse me, but I'd like to request more information about your question to provide an accurate response. Can you please elaborate?", 
    "Pardon my request, but I need more context to help you effectively. Could you kindly provide more details about your question?", 
    "I'm sorry, but I need more information to ensure I understand your question properly. Can you please give me more context?", 
    "My apologies, but to assist you accurately, I require more details about your question. Can you please provide more information?", 
    "I'm sorry, but I need more context to give you a well-informed answer. Would you be so kind as to elaborate on your question?", 
    "Excuse me, but I need additional information to provide you with the best answer. Can you please give me more context about your question?"]

@app.route('/query_cs', methods=['POST'])
def query():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    elif len(text.split()) < 3:
        time.sleep(2)
        return jsonify({'text': random.choice(too_short)}), 200
    else:
        response = agent_executor.run(text)
        return jsonify({'text': response})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)