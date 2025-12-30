from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

template = PromptTemplate(
    template="Give top 5 rules for game \n {game}",
    input_variables=["game"]
)

parser = StrOutputParser()

chain = template | model | parser
result = chain.invoke(
    {"game":"chess"}
)
print(result)

chain.get_graph().print_ascii()