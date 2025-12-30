from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
   
load_dotenv()

prompt1 = PromptTemplate(
    template="write a caption for instagram for a photo about {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="write a description for a photo about {topic}",
    input_variables=["topic"]
)
prompt3 = PromptTemplate(
    template="merge the caption-> {caption} and description-> {description} into a single paragraph",
    input_variables=["caption", "description"]
)
parser = StrOutputParser()

model = ChatOpenAI()

parallel_chain = RunnableParallel(
    caption=prompt1|model|parser,
    description=prompt2|model|parser
)

merge_chain = prompt3|model|parser

chain=parallel_chain | merge_chain

result=chain.invoke({"topic": "sunset at the beach"})
print(result)   
chain.get_graph().print_ascii()