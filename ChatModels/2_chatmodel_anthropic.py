from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()       
chat_model = ChatAnthropic(model_name="claude-2", temperature=0.7, max_tokens=10)
response = chat_model.invoke("Explain the theory of relativity in simple terms.")
print(response.content)