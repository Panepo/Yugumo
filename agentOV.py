from langchain.agents import AgentExecutor, StructuredChatAgent
from tools.weather import weather
from tools.wiki import wikipedia
from tools.device import device_query
from prompts.promptOv import PREFIX, SUFFIX, HUMAN_MESSAGE_TEMPLATE, FORMAT_INSTRUCTIONS
from models.modelOV import ov_llm

tools = [wikipedia, weather, device_query]

agent = StructuredChatAgent.from_llm_and_tools(
  ov_llm,
  tools,
  prefix=PREFIX,
  suffix=SUFFIX,
  human_message_template=HUMAN_MESSAGE_TEMPLATE,
  format_instructions=FORMAT_INSTRUCTIONS,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
