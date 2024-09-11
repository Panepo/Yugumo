from langchain.agents import create_openai_functions_agent, AgentExecutor
from tools.weather import weather
from tools.wiki import wikipedia
from tools.device import device_query
from prompts.promptGPT import prompt
from models.modelGPT import gpt_llm

tools = [wikipedia, weather, device_query]

agent = create_openai_functions_agent(
  llm=gpt_llm,
  prompt=prompt,
  tools=tools
)

agentExecutor = AgentExecutor(
  agent=agent,
  tools=tools,
  verbose=True
)

def process_chat(user_input, chat_history):
  response = agentExecutor.invoke({
    "input": user_input,
    "chat_history": chat_history
  })
  return response["output"]
