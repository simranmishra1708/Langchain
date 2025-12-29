from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    task = "text-generation",
    provider = "featherless-ai"
)

result = llm.invoke("What is the capital of India")

print(result)