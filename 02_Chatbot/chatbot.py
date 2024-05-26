from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

model = ChatOpenAI()
# model = ChatAnthropic(model="claude-3-sonnet-20240229")
# model = ChatVertexAI(model="gemini-pro", project="langchaintutorials-424420")

# result = model.invoke([
#         HumanMessage(content="Hi! I'm Bob"),
#         AIMessage(content="Hello Bob! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ])
# print(result)

# Message History

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable": {"session_id": "abc2"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
)

print(response)

# Используем историю и сессии abc2
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
# Помнит имя
print(response)

# Используем новую сессию abc3
config = {"configurable": {"session_id": "abc3"}}
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

# Не помнит имя
print(response)

##################################################
#               Prompt Templates
##################################################

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        # We will utilize MessagesPlaceholder to pass all the messages in.
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

response = chain.invoke({"messages": [HumanMessage(content="hi! I'm John Doe")]})

# Теперь вместо списка сообщений передаем словарь с ключом messages, содержащим список сообщений.
with_message_history = RunnableWithMessageHistory(chain, get_session_history)
config = {"configurable": {"session_id": "abc5"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Jim")],
    config=config,
)

print(response)


###################################################
#           Managing Conversation History
###################################################

# История может переполниться сообщениями, поэтому мы можем использовать функцию фильтрации,
# чтобы оставить только последние k сообщений.
def filter_messages(messages, k=10):
    return messages[-k:]


# Создаем цепочку, которая фильтрует сообщения перед передачей их в модель.
chain = (
        RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
        | prompt
        | model
)
# Допустим у нас есть история сообщений, которая содержит 10 сообщений.
messages = [
    HumanMessage(content="hi! I'm Martin"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# Теперь спрашиваем имя, которое упомянуто в первом сообщении.
# Поскольку у нас есть фильтрация, модель увидит только последние 10 сообщений.
# Сообщение "hi! I'm Martin" будет проигнорировано.
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    }
)

print(response)  # Response: I'm sorry, I don't know your name.

# Теперь спрашиваем имя, которое упомянуто во втором сообщении.
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my fav ice cream")],
        "language": "English",
    }
)

print(response)  # Response: Your favorite ice cream flavor is vanilla.

# Теперь обернем цепочку в RunnableWithMessageHistory, чтобы использовать историю сообщений.
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc20"}}
# Теперь в истории еще больше сообщений.
response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)

print(response)  # Response: I'm sorry, I don't know your name.

# Второе сообщение в истории о мороженом также буде проигнорировано.
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="whats my favorite ice cream?")],
        "language": "English",
    },
    config=config,
)

print(response)  # Response: I'm sorry, I don't know your favorite ice cream flavor.


###################################################
#           Streaming
###################################################

# Мы можем использовать метод stream для пошагового выполнения цепочки.
config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "Russian",
    },
    config=config,
):
    print(r.content, end="|")