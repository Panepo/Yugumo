from langchain.agents import create_openai_functions_agent, create_structured_chat_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from tools.weather import weather
from tools.wiki import wikipedia

from dotenv import load_dotenv
import os

load_dotenv()

backend = os.getenv("BACKEND")
verbose = True if os.getenv("VERBOSE") == "true" else False

tools = [wikipedia, weather]

prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "Respond to the human as helpfully and accurately as possible. You have access to the following tools: {tools}"),
    ("human", "{input}")
  ]
)

if backend == "cuda" or backend == "cpu":
  from models.modelHF import hf_llm
  agent = create_structured_chat_agent(
    llm=hf_llm,
    prompt=prompt,
    tools=tools
  )
elif backend == "gpt":
  from models.modelGPT import gpt_llm
  agent = create_openai_functions_agent(
    llm=gpt_llm,
    prompt=prompt,
    tools=tools
  )
else:
  raise ValueError(f"Unknown backend: {backend}")

agentExecutor = AgentExecutor(
  agent=agent,
  tools=tools,
  verbose=verbose
)

def process_chat(user_input, chat_history):
  response = agentExecutor.invoke({
    "input": user_input,
    "chat_history": chat_history
  })
  return response["output"]
