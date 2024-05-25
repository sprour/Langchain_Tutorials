from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langserve import RemoteRunnable

load_dotenv()

openai = ChatOpenAI()
claude = ChatAnthropic(model="claude-3-sonnet-20240229")
model = ChatVertexAI(model="gemini-pro", project="langchaintutorials-424420")

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi! My name is Sergei. I am a software engineer."),
]

# result = gemini.invoke(messages)
parser = StrOutputParser()
# output = parser.invoke(result)
# print(output)

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# result = prompt_template.invoke({"language": "italian", "text": "hi"})

# ChatPromptValue(messages=[SystemMessage(content='Translate the following into italian:'), HumanMessage(content='hi')])

# We can easily create the chain using the | operator.
# The | operator is used in LangChain to combine the elements together
chain = prompt_template | model | parser

# result = chain.invoke({"language": "italian", "text": "hi"})

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
result = remote_chain.invoke({"language": "italian", "text": "hi"})

print(result)
