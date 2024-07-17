from langchain.agents import AgentExecutor, StructuredChatAgent
from tools.math import add, multiply, exponentiate
from tools.weather import weather
from tools.wiki import wikipedia
from prompts.promptTool import PREFIX, SUFFIX, HUMAN_MESSAGE_TEMPLATE, FORMAT_INSTRUCTIONS
from models.ovmodel import ov_llm

tools = [wikipedia, weather]

#approval_handler = HumanApprovalCallbackHandler()

agent = StructuredChatAgent.from_llm_and_tools(
    ov_llm,
    tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    human_message_template=HUMAN_MESSAGE_TEMPLATE,
    format_instructions=FORMAT_INSTRUCTIONS,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
