from langchain.agents import AgentExecutor, StructuredChatAgent
from tools.math import add, multiply, exponentiate
from tools.weather import weather
from tools.wiki import wikipedia
from tools.license import license_plate_query
from prompt import PREFIX, SUFFIX, HUMAN_MESSAGE_TEMPLATE, FORMAT_INSTRUCTIONS
from models.ovmodel import ov_llm
from models.hfmodel import hf_llm

tools = [wikipedia, weather, license_plate_query, multiply, add, exponentiate,]

agent = StructuredChatAgent.from_llm_and_tools(
    hf_llm,
    tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    human_message_template=HUMAN_MESSAGE_TEMPLATE,
    format_instructions=FORMAT_INSTRUCTIONS,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
