import os
from agent import agent_executor

os.environ["TF_ENABLE_ONEDNN_OPTS"] = 0

while 1:
    print("================================================")
    text = input("Please say something: ")
    # agent_executor.invoke({"input": {text}})

    for new_text in agent_executor.stream(
        {"input": {text}},
    ):
        if "output" in new_text.keys():
            print(new_text["output"])
