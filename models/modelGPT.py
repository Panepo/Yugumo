import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

gpt_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
