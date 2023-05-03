from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import OpenAI, LLMChain
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import List, Union
# from elevenlabs import generate, play
from llama_index import GPTSimpleVectorIndex, LLMPredictor
from langchain.utilities import SerpAPIWrapper
import pandas as pd
import re

from dotenv import load_dotenv
load_dotenv()

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
            source_text = f"\nSource:\nTitle: {title}\nConfidence: {source_node.score:.3f}\nURL: {docid_to_url[doc_id]}"
            texts.append(source_text)
        return "\n".join(texts)


template = """You are a friendly Roku customer support agent. People who talk with you might not be tech-savvy; you can break down the instructions into smaller, more manageable steps. Instead of providing a long list of actions to take, you could break down each step and explain it thoroughly in short, simple sentences. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take should be one of [{tool_names}], but prioritize the "CS Vector Index." Please include the URL of the source you used.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. Include the source from the Observation.

Begin! Remember to be friendly and explain things thoroughly with simple language.

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

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)


while True:
    text_input = input("User: ")
    response = agent_executor.run(input=text_input)
    print(f'Agent: {response}')

