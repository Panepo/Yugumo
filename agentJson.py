from langchain.agents import AgentExecutor, create_json_chat_agent
from tools.weather import weather
from tools.wiki import wikipedia
from tools.driver import driver_query
from prompts.promptJson import PROMPT
from models.ovmodel import ov_llm

tools = [wikipedia, weather, driver_query]

agent = create_json_chat_agent(ov_llm, tools, PROMPT)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
