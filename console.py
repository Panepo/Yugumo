from agent import agent_executor

while 1:
    print("================================================")
    text = input("Please say something: ")
    agent_executor.invoke({"input": {text}})
