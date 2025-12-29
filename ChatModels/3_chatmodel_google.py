from langchain_google_genai import ChatGoogleGenAI
from dotenv import load_dotenv
load_dotenv()
chat_model = ChatGoogleGenAI(model_name="gemini-1.5-turbo", temperature=0.7, max_tokens=10)
response = chat_model.invoke("Summarize the plot of 'Inception' in one sentence.")
print(response.content)