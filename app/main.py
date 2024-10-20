import time, re
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.jgmancilla.com"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############################################################################################################

class Question(BaseModel):
    question: str

urls = {
    "calvinist parrot": "https://www.jgmancilla.com/calvinist-parrot",
    "customer support bot": "https://www.jgmancilla.com/customer-support-bot-project",
    "design research": "https://www.jgmancilla.com/design-research",
    "ml portfolio": "https://www.jgmancilla.com/ml-portfolio",
    "research librarian": "https://www.jgmancilla.com/research-librarian",
    "research operations": "https://www.jgmancilla.com/research-ops",
    "research portfolio": "https://www.jgmancilla.com/research-portfolio",
    "resume": "https://www.jgmancilla.com/resume/",
    "survey report generator": "https://www.jgmancilla.com/modular-survey-report-generator",
    "user research": "https://www.jgmancilla.com/user-research",
    "papers": "https://www.jgmancilla.com/research-papers"
}

def parse_annotations(annotations, assistant_response):
    for i in annotations:
        match = re.search(r'†(.*?).txt】', i.text)
        if match:
            filename = match.group(1)
            url = urls.get(filename, 'URL not found')
            # Include target="_blank" and rel="noopener noreferrer"
            link_html = f' (<a href="{url}" target="_blank" rel="noopener noreferrer">{filename.title()}</a>)'
            assistant_response = assistant_response.replace(i.text, link_html)
    return assistant_response

@app.post('/representative')
def representative(input: Question):
    assistant_id = 'asst_oSrgDfpLyLtV64PaOCvTMtKm'  # Replace with your assistant ID

    try:
        # Step 1: Create a new thread
        thread = client.beta.threads.create()

        # Step 2: Add the user's message to the thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=input.question
        )

        # Step 3: Create and poll the run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Step 4: Check if the run was successful
        while run.status == "queued" or run.status == "in_progress":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(1)
        
        assistant_response = None

        # Retrieve the assistant's response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        assistant_response = messages.data[0].content[0].text.value
        annotations = messages.data[0].content[0].text.annotations

        assistant_response = parse_annotations(annotations, assistant_response)

        if assistant_response:
            return {'response': assistant_response}
        else:
            return {'response': 'No assistant response found.'}

    except Exception as e:
        print(f"Error: {e}")
        return {'response': 'An error occurred while processing your request.'}