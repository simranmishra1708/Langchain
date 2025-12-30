from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch,RunnableLambda,RunnableParallel
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI()

parser1= StrOutputParser()

class Review(BaseModel):
    sentiment:Literal["positive","negative","neutral"] =Field(description="Sentiment of the review")

parser2= PydanticOutputParser(pydantic_object=Review)

prompt1 = PromptTemplate(
    template="Extract sentiment from review: {review}\n{format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

chain1=prompt1|model|parser2

prompt2= PromptTemplate(
    template="write an appropriate response on positive reviews: {review}",
    input_variables=["review"]
)

prompt3= PromptTemplate(
    template="write an appropriate response on negative reviews: {review}",
    input_variables=["review"]
)

prompt4= PromptTemplate(
    template="write an appropriate response on neutral reviews: {review}",
    input_variables=["review"]
)

branch_chain=RunnableBranch(
    (lambda x: x.sentiment=="positive", prompt2|model|parser1),
    (lambda x: x.sentiment=="negative", prompt3|model|parser1),
    (lambda x: x.sentiment=="neutral", prompt4|model|parser1),
    RunnableLambda(lambda x: "sentiment not recognized")
)
chain= chain1 | branch_chain
result=chain.invoke(
    {"review":"I absolutely hate the new features of the product! It has significantly worsened my productivity and the user experience is pathetic."}
)   
print(result)
# chain.get_graph().print_ascii()