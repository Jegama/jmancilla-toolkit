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
    allow_origins=[
        "https://www.jgmancilla.com",
        "https://jgmancilla.com",
        "http://localhost:3000",
        "https://mancillaconsulting.com",
        "https://www.mancillaconsulting.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############################################################################################################

class Question(BaseModel):
    question: str

urls = {
    "file-YzIK2M7nHYolNPOQSNDDmqIo": {
        "name": "Calvinist Parrot",
        "url": "https://www.jgmancilla.com/calvinist-parrot"
    },
    "file-JBOt11edH7bhRmwmLFxequke": {
        "name": "Customer Support Bot",
        "url": "https://www.jgmancilla.com/customer-support-bot-project"
    },
    "file-FpU75ViRtXUfcdrIxIxUVKjO": {
        "name": "Design Research",
        "url": "https://www.jgmancilla.com/design-research"
    },
    "file-2qlNapRedI07LtDlB4XrXok3": {
        "name": "ML Portfolio",
        "url": "https://www.jgmancilla.com/ml-portfolio"
    },
    "file-9jVmZj2fzRJblZtXYTaViFXE": {
        "name": "Research Librarian",
        "url": "https://www.jgmancilla.com/research-librarian"
    },
    "file-uPsZ8lcTIrv15GcyNvot0WYp": {
        "name": "Research Ops",
        "url": "https://www.jgmancilla.com/research-ops"
    },
    "file-coZ6NmnlSKBhPDWdIlb9d3Q2": {
        "name": "Research Portfolio",
        "url": "https://www.jgmancilla.com/research-portfolio"
    },
    "file-vqNHbAjvyO8CGkg0YfRZ2biS": {
        "name": "Resume",
        "url": "https://www.jgmancilla.com/resume"
    },
    "file-DmXVTCpRpgPtoHpnNWfexxVK": {
        "name": "Survey Report Generator",
        "url": "https://www.jgmancilla.com/modular-survey-report-generator"
    },
    "file-M7Xah9KfmfFWN5yyVKWkQpVb": {
        "name": "User Research",
        "url": "https://www.jgmancilla.com/user-research"
    },
    "file-Crej1pUIVVod5DvmIF1VguSG": {
        "name": "Research Papers",
        "url": "https://www.jgmancilla.com/research-papers"
    }
}

guarding_sys_prompt = """You are tasked with analyzing the user's question and categorizing it into one of the following categories:

1. **Machine Learning Experience**
2. **Research Projects and Publications**
3. **User Experience Research (UXR)**
4. **Quantitative and Mixed-Methods Skills**
5. **Skillset and Competencies**
6. **Non-Related**

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
    {"role": "user", "content": "What are his top 3 competencies?"},
    {"role": "assistant", "content": "{'reformatted_question': 'What are Jesús Mancilla\'s skillset and competencies?','category': 'Skillset and Competencies'}"},
    {"role": "user", "content": "Would he be a good fit for a Lead AI Engineer?"},
    {"role": "assistant", "content": "{'reformatted_question': 'Would Jesús Mancilla be a good fit for a Lead AI Engineer position?','category': 'Machine Learning Experience'}"}
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
    lines = assistant_response.split("\n")

    new_lines = []
    for line in lines:
        replaced_file_ids = set()
        # Gather annotations that appear in this line
        line_annotations = []
        for ann in annotations:
            index = line.find(ann.text)
            if index != -1:  # annotation snippet is in this line
                line_annotations.append((index, ann))
        
        # Sort them so we replace in order of appearance
        line_annotations.sort(key=lambda x: x[0])
        
        output_line = ""
        last_pos = 0
        
        for (found_index, ann) in line_annotations:
            file_id = ann.file_citation.file_id
            # If this annotation has already been replaced for this line, or if the snippet doesn't appear anymore, skip it.
            if file_id in replaced_file_ids:
                # skip it (remove it), so just slice out the snippet
                output_line += line[last_pos:found_index]
                # jump over the snippet
                last_pos = found_index + len(ann.text)
            else:
                # do a single replacement with the link
                filename = urls[file_id]["name"]
                file_url = urls[file_id]["url"]
                
                output_line += line[last_pos:found_index]
                output_line += f" ([{filename}]({file_url}))"
                replaced_file_ids.add(file_id)
                last_pos = found_index + len(ann.text)
        
        # Add any trailing text after the last annotation
        output_line += line[last_pos:]
        new_lines.append(output_line)

    # Join it all back
    assistant_response = "\n".join(new_lines)

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