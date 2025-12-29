from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  

from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-5.2", temperature=0.7)

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while (True):
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break

    chat_history.append(HumanMessage(content=user_input))

    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")

print("Chat ended.", chat_history)