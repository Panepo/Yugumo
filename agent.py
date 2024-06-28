from langchain.agents import AgentExecutor, StructuredChatAgent
from tools.math import add, multiply, exponentiate
from tools.weather import weather
from tools.ecr import ecr
from tools.wiki import wikipedia
from tools.device import device_list
from prompts.openvino import PREFIX, SUFFIX, HUMAN_MESSAGE_TEMPLATE, FORMAT_INSTRUCTIONS
from model import ov_llm

tools = [wikipedia, weather, ecr, device_list, multiply, add, exponentiate,]

agent = StructuredChatAgent.from_llm_and_tools(
    ov_llm,
    tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    human_message_template=HUMAN_MESSAGE_TEMPLATE,
    format_instructions=FORMAT_INSTRUCTIONS,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
