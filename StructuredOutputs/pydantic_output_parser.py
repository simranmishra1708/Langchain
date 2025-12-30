from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City where the person lives")



# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)    

parser= PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate name, age and city of a fictional person from {place}\n{format_instructions}",
    input_variables=['place'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt=template.invoke({"place":"Canada"})
# result = model.invoke(prompt)
# parsed_output = parser.parse(result.content)
# print(parsed_output)

chain = template | model | parser
result =chain.invoke({"place":"Canada"})
print(result)