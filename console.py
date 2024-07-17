from agentJson import agent_executor

while 1:
    print("================================================")
    text = input("Please say something: ")
    # agent_executor.invoke({"input": {text}})

    for new_text in agent_executor.stream(
        {"input": {text}},
    ):
        if "output" in new_text.keys():
            print("================================================")
            print(new_text["output"])
