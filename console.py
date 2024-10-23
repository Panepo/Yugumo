
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import time

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "openvino":
  from agentOV import process_chat
elif backend == "cuda" or backend == "cpu":
  from agent import process_chat
else:
  raise ValueError(f"Unknown backend: {backend}")

chat_history = []

while 1:
  start_time = time.process_time()
  print("================================================")
  text = input("Please say something: ")
  response = process_chat(text, chat_history)
  chat_history.append(HumanMessage(content=text))
  chat_history.append(AIMessage(content=response))
  print("================================================")
  print(response)
  end_time = time.process_time()
  elapsed_time = end_time - start_time
  print(f"Elapsed time: {elapsed_time} seconds")
