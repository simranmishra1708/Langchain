from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv  
load_dotenv()
model = ChatOpenAI()
prompt1 = PromptTemplate(
    template="give me a brief description of {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Summarize the following text in one sentence: {text}",
    input_variables=["text"]
)
parser = StrOutputParser()  
chain = prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({"topic": "cricket"})
print(result)
chain.get_graph().print_ascii()