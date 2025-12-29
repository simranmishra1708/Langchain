from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=4)  
documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It enables developers to build applications that can understand and generate human-like text.",
    "LangChain provides tools for prompt management, memory, and integration with external data sources."
]
doc_embeddings = embedding.embed_documents(documents)   
query = "hownlangchain helps in building models?"
query_embedding = embedding.embed_query(query)

score = cosine_similarity([query_embedding], doc_embeddings)[0]
index,score=sorted(list(enumerate(score)),key=lambda x: x[1])[-1]
print(query)
print(documents[index])