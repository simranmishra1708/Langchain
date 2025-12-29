from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.7, "max_new_tokens": 10},
)
result = llm.invoke("What is the capital of India") 
print(result)