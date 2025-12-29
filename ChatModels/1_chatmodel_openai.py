from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model_name="gpt-5", temperature=0.7, max_tokens=10)
response = chat_model.invoke("Write some insteresting facts about indian name simran?")
print(response.content)