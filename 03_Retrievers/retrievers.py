from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from typing import List

load_dotenv()

##################################################
#                   Documents
##################################################

# Создаем список документов
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

##################################################
#                   Vector stores
##################################################

# Вызывая метод from_documents, мы можем создать векторное хранилище из документов
vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

# Поиск по векторному хранилищу
# На первое место выведется документ с кошками
response = vectorstore.similarity_search("Cat")
print(response)

# response = await vectorstore.asimilarity_search("dog")
# print(response)

# Поиск по векторному хранилищу с возвращением оценки
# На первое место выведется документ с собаками
response = vectorstore.similarity_search_with_score("Dog")
print(response)

# Создаем вектор для запроса
embedding = OpenAIEmbeddings().embed_query("Rabbit")
# Поиск по векторному хранилищу с вектором запроса
# На первое место выведется документ с кроликами
response = vectorstore.similarity_search_by_vector(embedding)
print(response)

##################################################
#                  Retrievers
##################################################

# Создаем ретриевер, который будет возвращать 1 результат
retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

# Поиск по векторному хранилищу с ретриевером batch, который позволяет передавать несколько запросов
response = retriever.batch(["cat", "shark"])
print(response)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

response = retriever.batch(["Bird", "shark"])

print(response)

##################################################
#                   Prompt Templates
##################################################

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""
# Создаем шаблон для чат-промпта, где ""human" - это имя пользователя, а message - это его сообщение
prompt = ChatPromptTemplate.from_messages([("human", message)])

# Создаем цепочку, которая будет выполнять ретриевер и промпт
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

# Вызываем цепочку с запросом.
response = rag_chain.invoke("tell me about cats")

print(response.content)  # Output: Cats are independent pets that often enjoy their own space.
