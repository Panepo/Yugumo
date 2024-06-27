from langchain.agents import AgentExecutor, StructuredChatAgent
from tools.math import add, multiply, exponentiate
from tools.weather import weather
from template import PREFIX, SUFFIX, HUMAN_MESSAGE_TEMPLATE, FORMAT_INSTRUCTIONS
from model import ov_llm

tools = [multiply, add, exponentiate, weather]

agent = StructuredChatAgent.from_llm_and_tools(
    ov_llm,
    tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    human_message_template=HUMAN_MESSAGE_TEMPLATE,
    format_instructions=FORMAT_INSTRUCTIONS,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke({"input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"})

while (1):
  print("================================================")
  text = input("Please say something: ")
  agent_executor.invoke({"input": {text}})
