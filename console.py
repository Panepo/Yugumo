from agentGPT import process_chat
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

while 1:
  print("================================================")
  text = input("Please say something: ")
  response = process_chat(text, chat_history)
  chat_history.append(HumanMessage(content=text))
  chat_history.append(AIMessage(content=response))
  print("================================================")
  print(response['output'])
