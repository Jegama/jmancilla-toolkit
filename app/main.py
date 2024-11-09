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

guarding_sys_prompt = """You are tasked with analyzing the user's question and categorizing it into one of the following categories:

1. **Machine Learning Experience**
2. **Research Projects and Publications**
3. **User Experience Research (UXR)**
4. **Quantitative and Mixed-Methods Skills**
5. **Non-Related**

**Instructions:**

- **Reformat the User's Question:** Rewrite the question for clarity and focus, centering it around Jesús Mancilla's experience.
- **Assign a Category:** Choose the most appropriate category from the list above that fits the user's question.
- If the question does not pertain to any of the first four categories related to Jesús Mancilla's work, assign it the category **"Non-Related"**.

**Important:**

- We have a refusal system that utilizes the "Non-Related" category to redirect the conversation back to Jesús Mancilla's experience.
- Always provide your response strictly in the following JSON format:

```json
{
    "reformatted_question": "Your reformatted question here",
    "category": "Assigned category here"
}
```

Do not include any additional text outside of the JSON response."""

n_shoot_examples = [
    {"role": "user", "content": "Can you tell me about his work in machine learning?"},
    {"role": "assistant", "content": "{'reformatted_question': 'What is Jesús Mancilla\'s experience in machine learning?','category': 'Machine Learning Experience'}"},
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "{'reformatted_question': 'Non Applicable','category': 'Non-Related'}"},
    {"role": "user", "content": "What research projects has Jesús been involved in?"},
    {"role": "assistant", "content": "{'reformatted_question': 'What are the research projects and publications of Jesús Mancilla?','category': 'Research Projects and Publications'}"},
    {"role": "user", "content": "Can you help me with my homework?"},
    {"role": "assistant", "content": "{'reformatted_question': 'Non Applicable','category': 'Non-Related'}"},
    {"role": "user", "content": "Does he has experience with user experience research?"},
    {"role": "assistant", "content": "{'reformatted_question': 'What is Jesús Mancilla\'s experience in user experience research (UXR)?','category': 'User Experience Research (UXR)'}"},
    {"role": "user", "content": "What quantitative methods is Jesús skilled in?"},
    {"role": "assistant", "content": "{'reformatted_question': 'What are Jesús Mancilla\'s quantitative and mixed-methods skills?','category': 'Quantitative and Mixed-Methods Skills'}"},
    {"role": "user", "content": "What's the best place to eat in town?"},
    {"role": "assistant", "content": "{'reformatted_question': 'Non Applicable','category': 'Non-Related'}"},
]

def verification_step(messages_list, model_to_use = "gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model_to_use,
        messages=messages_list,
        response_format={ "type": "json_object" },
        temperature = 0
    )
    temp = response.choices[0].message.content
    return eval(temp)


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
    assistant_id = 'asst_oSrgDfpLyLtV64PaOCvTMtKm' 

    try:
        # Step 1: Verify the user's question
        step_1_prompt = [{"role": "system", "content": guarding_sys_prompt}] + n_shoot_examples + [{"role": "user", "content": input.question}]
        verification = verification_step(step_1_prompt)

        if verification['category'] == 'Non-Related':
            return {'response': 'I am sorry but I am only able to provide information about Jesús Mancilla\'s experience. Please ask a question related to his work.'}
        
        reformatted_question = verification['reformatted_question']

        # Step 2: Create a new thread
        thread = client.beta.threads.create()

        # Step 3: Add the user's message to the thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=reformatted_question
        )

        # Step 4: Create and poll the run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Step 5: Check if the run was successful
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