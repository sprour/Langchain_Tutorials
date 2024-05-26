---
sidebar_position: 1
---

# Построение чат-бота

## Обзор

Мы рассмотрим пример того, как проектировать и реализовывать чат-бот, работающий на основе большой языковой модели (LLM).
Этот чат-бот будет способен вести диалог и запоминать предыдущие взаимодействия.

Обратите внимание, что созданный нами чат-бот будет использовать языковую модель только для ведения разговора.
Существует несколько других связанных концепций, которые могут вас заинтересовать:

- [Разговорный RAG](/docs/tutorials/qa_chat_history): Реализация чат-бота с использованием внешнего источника данных.
- [Агенты](/docs/tutorials/agents): Создание чат-бота, который может выполнять действия.

Данный урок освещает основы, которые будут полезны для этих двух более продвинутых тем. Но вы можете перейти непосредственно к ним, если хотите.

## Концепции

Вот несколько основных компонентов, с которыми мы будем работать:

- [`Chat Models`](/docs/concepts/#chat-models). Интерфейс чат-бота основан на сообщениях, а не на простом тексте, поэтому для него лучше подходят Chat Models, а не текстовые LLM.
- [`Prompt Templates`](/docs/concepts/#prompt-templates), которые упрощают процесс создания запросов, комбинируя стандартные сообщения, пользовательский ввод, историю чата и (по желанию) дополнительный извлеченный контекст.
- [`Chat History`](/docs/concepts/#chat-history), позволяющий чат-боту "запоминать" прошлые взаимодействия и учитывать их при ответе на последующие вопросы.
- Отладка и трассировка вашего приложения с помощью [LangSmith](/docs/concepts/#langsmith).

Мы рассмотрим, как объединить эти компоненты для создания мощного разговорного чат-бота.

## Настройка

### Jupyter Notebook

Данное руководство (и большинство других руководств в документации) использует [Jupyter notebooks](https://jupyter.org/) и предполагает, что вы тоже используете их. Jupyter notebooks идеально подходят для изучения работы с системами LLM, поскольку часто возникают проблемы (неожиданный вывод, нерабочий API и т. д.), а работа с руководствами в интерактивной среде позволяет лучше понять их.

Этот и другие уроки удобнее всего запускать в Jupyter notebook. Инструкции по установке можно найти [здесь](https://jupyter.org/install).

### Установка

Чтобы установить LangChain, выполните:

```{=mdx}
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import CodeBlock from \"@theme/CodeBlock\";

<Tabs>
<TabItem value=\"pip\" label=\"Pip\" default>
<CodeBlock language=\"bash\">pip install langchain</CodeBlock>
</TabItem>
<TabItem value=\"conda\" label=\"Conda\">
<CodeBlock language=\"bash\">conda install langchain -c conda-forge</CodeBlock>
</TabItem>
</Tabs>

```

Для получения более подробной информации ознакомьтесь с нашим [Руководством по установке](/docs/how_to/installation).

### LangSmith

Многие приложения, которые вы создаете с помощью LangChain, будут содержать несколько шагов с множественными вызовами LLM.
По мере того, как эти приложения становятся все более сложными, становится крайне важно иметь возможность просматривать, что именно происходит внутри вашей цепочки или агента.
Лучший способ сделать это - с помощью [LangSmith](https://smith.langchain.com).

После регистрации по указанной выше ссылке убедитесь, что вы установили свои переменные среды, чтобы начать запись трассировок:

```shell
export LANGCHAIN_TRACING_V2=\"true\"
export LANGCHAIN_API_KEY=\"...\"
```

Или, если вы работаете с Jupyter notebook, вы можете установить их с помощью:

```python
import getpass
import os

os.environ[\"LANGCHAIN_TRACING_V2\"] = \"true\"
os.environ[\"LANGCHAIN_API_KEY\"] = getpass.getpass()
```

## Быстрый запуск

Сначала давайте научимся использовать языковую модель самостоятельно. LangChain поддерживает множество различных языковых моделей, которые вы можете использовать взаимозаменяемо - выберите ту, которую вы хотите использовать ниже!

```{=mdx}
import ChatModelTabs from \"@theme/ChatModelTabs\";

<ChatModelTabs openaiParams={`model=\"gpt-3.5-turbo\"`} />
```

```python
# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model=\"gpt-3.5-turbo\")
```

Сначала давайте воспользуемся моделью напрямую. `ChatModel` - это экземпляры "выполняемых" элементов LangChain, что означает, что они предоставляют стандартный интерфейс для взаимодействия с ними. Для простого вызова модели мы можем передать список сообщений в метод `.invoke`.

```python
from langchain_core.messages import HumanMessage

model.invoke([HumanMessage(content=\"Hi! I'm Bob\")])
```

Модель сама по себе не имеет понятия о состоянии. Например, если вы зададите последующий вопрос:

```python
model.invoke([HumanMessage(content=\"What's my name?\")])
```

Давайте посмотрим на [LangSmith trace](https://smith.langchain.com/public/5c21cb92-2814-4119-bae9-d02b8db577ac/r)
в качестве примера.

Мы видим, что она не принимает во внимание предыдущий ход беседы и не может ответить на вопрос.
Это создает ужасный опыт взаимодействия с чат-ботом!

Чтобы решить эту проблему, нам нужно передать модели всю историю разговора. Давайте посмотрим, что произойдет, если мы это сделаем:

```python
from langchain_core.messages import AIMessage

model.invoke(
[
HumanMessage(content=\"Hi! I'm Bob\"),
AIMessage(content=\"Hello Bob! How can I assist you today?\"),
HumanMessage(content=\"What's my name?\"),
]
)
```

И теперь мы видим хороший ответ!

Это основная идея, лежащая в основе способности чат-бота взаимодействовать в режиме диалога.
Итак, как нам лучше всего реализовать это?

## История сообщений

Мы можем использовать класс Message History, чтобы обернуть нашу модель и сделать ее состоятельной.
Этот класс будет отслеживать входные и выходные данные модели и хранить их в некотором хранилище данных.
В будущем взаимодействия будут загружать эти сообщения и передавать их в цепочку в качестве части входных данных.
Давайте посмотрим, как это сделать!

Сначала давайте убедимся, что мы установили `langchain-community`, так как мы будем использовать интеграцию в нем для хранения истории сообщений.

```python
# ! pip install langchain_community
```

После этого мы можем импортировать соответствующие классы и настроить нашу цепочку, которая обертывает модель и добавляет эту историю сообщений. Ключевой частью здесь является функция, которую мы передаем в качестве `get_session_history`. Ожидается, что эта функция будет принимать `session_id` и возвращать объект Message History. Этот `session_id` используется для различения отдельных бесед и должен передаваться в качестве части конфигурации при вызове новой цепочки (мы покажем, как это сделать).

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
if session_id not in store:
store[session_id] = ChatMessageHistory()
return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)
```

Теперь нам нужно создать `config`, который мы будем передавать в выполняемый объект каждый раз. Эта конфигурация содержит информацию, которая не является частью входных данных напрямую, но все равно полезна. В данном случае мы хотим включить `session_id`. Это должно выглядеть так:

```python
config = {"configurable": {"session_id": "abc2"}}
```

```python
response = with_message_history.invoke(
[HumanMessage(content=\"Hi! I'm Bob\")],
config=config,
)

response.content
```

```python
response = with_message_history.invoke(
[HumanMessage(content=\"What's my name?\")],
config=config,
)

response.content
```

Отлично! Наш чат-бот теперь помнит о нас. Если мы изменим конфигурацию, чтобы ссылаться на другой `session_id`, мы увидим, что он начинает разговор заново.

```python
config = {"configurable": {"session_id": "abc3"}}

response = with_message_history.invoke(
[HumanMessage(content=\"What's my name?\")],
config=config,
)

response.content
```

Однако мы всегда можем вернуться к исходной беседе (так как мы сохраняем ее в базе данных).

```python
config = {"configurable": {"session_id": "abc2"}}

response = with_message_history.invoke(
[HumanMessage(content=\"What's my name?\")],
config=config,
)

response.content
```

Вот как мы можем поддерживать чат-бота, который ведет беседы со многими пользователями!

Сейчас мы просто добавили простой слой сохранения данных вокруг модели. Мы можем сделать ее более сложной и персонализированной, добавив шаблон запроса.

## Шаблоны запросов

Шаблоны запросов помогают преобразовывать необработанную пользовательскую информацию в формат, с которым может работать LLM. В данном случае необработанным пользовательским вводом является просто сообщение, которое мы передаем LLM. Давайте сделаем это немного сложнее. Сначала добавим системное сообщение с некоторыми пользовательскими инструкциями (но по-прежнему принимая сообщения в качестве входных данных). Затем добавим больше входных данных, помимо просто сообщений.

Сначала добавим системное сообщение. Для этого мы создадим ChatPromptTemplate. Мы будем использовать `MessagesPlaceholder` для передачи всех сообщений.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"You are a helpful assistant. Answer all questions to the best of your ability.",
),
MessagesPlaceholder(variable_name="messages"),
]
)

chain = prompt | model
```

Обратите внимание, что это немного меняет тип входных данных - вместо того, чтобы передавать список сообщений, мы теперь передаем словарь с ключом `messages`, где этот ключ содержит список сообщений.

```python
response = chain.invoke({"messages": [HumanMessage(content=\"hi! I'm bob\")]})

response.content
```

Теперь мы можем обернуть это в тот же объект Messages History, что и раньше.

```python
with_message_history = RunnableWithMessageHistory(chain, get_session_history)
```

```python
config = {"configurable": {"session_id": "abc5"}}
```

```python
response = with_message_history.invoke(
[HumanMessage(content=\"Hi! I'm Jim\")],
config=config,
)

response.content
```

```python
response = with_message_history.invoke(
[HumanMessage(content=\"What's my name?\")],
config=config,
)

response.content
```

Отлично! Давайте теперь сделаем наш запрос немного сложнее. Предположим, что шаблон запроса теперь выглядит примерно так:

```python
prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
),
MessagesPlaceholder(variable_name="messages"),
]
)

chain = prompt | model
```

Обратите внимание, что мы добавили новый входной параметр `language` в запрос. Теперь мы можем вызвать цепочку и передать язык по своему выбору.

```python
response = chain.invoke(
{"messages": [HumanMessage(content=\"hi! I'm bob\")], "language": "Spanish"}
)

response.content
```

Теперь давайте обернем эту более сложную цепочку в класс Message History. На этот раз, поскольку во входных данных есть несколько ключей, нам нужно указать правильный ключ для сохранения истории чата.

```python
with_message_history = RunnableWithMessageHistory(
chain,
get_session_history,
input_messages_key="messages",
)
```

```python
config = {"configurable": {"session_id": "abc11"}}
```

```python
response = with_message_history.invoke(
{"messages": [HumanMessage(content=\"hi! I'm todd\")], "language": "Spanish"},
config=config,
)

response.content
```

```python
response = with_message_history.invoke(
{"messages": [HumanMessage(content=\"whats my name?\")], "language": "Spanish"},
config=config,
)

response.content
```

Чтобы лучше понять, что происходит в системе, посмотрите [эту LangSmith trace](https://smith.langchain.com/public/f48fabb6-6502-43ec-8242-afc352b769ed/r).

## Управление историей разговора

Одна из важных концепций, которую необходимо понять при создании чат-ботов, - это управление историей разговора. Если не управлять ею, список сообщений будет неограниченно расти и потенциально превысит контекстное окно LLM. Поэтому важно добавить шаг, который ограничивает размер передаваемых сообщений.

**Важно, что вам нужно сделать это ДО шаблона запроса, но ПОСЛЕ загрузки предыдущих сообщений из Message History.**

Мы можем сделать это, добавив простой шаг перед запросом, который модифицирует ключ `messages` соответствующим образом, а затем обернуть эту новую цепочку в класс Message History. Сначала давайте определим функцию, которая будет модифицировать передаваемые сообщения. Давайте сделаем так, чтобы она выбирала `k` самых последних сообщений. Затем мы можем создать новую цепочку, добавив ее в начале.

```python
from langchain_core.runnables import RunnablePassthrough


def filter_messages(messages, k=10):
return messages[-k:]


chain = (
RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
| prompt
| model
)
```

Давайте теперь проверим это! Если мы создадим список сообщений, превышающий 10 сообщений, мы увидим, что он больше не помнит информацию в первых сообщениях.

```python
messages = [
HumanMessage(content="hi! I'm bob"),
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
```

```python
response = chain.invoke(
{
"messages": messages + [HumanMessage(content="what's my name?")],
"language": "English",
}
)
response.content
```

Но если мы спросим о информации, которая находится в последних десяти сообщениях, она все еще помнит ее.

```python
response = chain.invoke(
{
"messages": messages + [HumanMessage(content="what's my fav ice cream")],
"language": "English",
}
)
response.content
```

Теперь давайте обернем это в Message History.

```python
with_message_history = RunnableWithMessageHistory(
chain,
get_session_history,
input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc20"}}
```

```python
response = with_message_history.invoke(
{
"messages": messages + [HumanMessage(content="whats my name?")],
"language": "English",
},
config=config,
)

response.content
```

Теперь в истории чата появилось два новых сообщения. Это означает, что еще больше информации, которая раньше была доступна в истории нашего разговора, теперь недоступна!

```python
response = with_message_history.invoke(
{
"messages": [HumanMessage(content="whats my favorite ice cream?")],
"language": "English",
},
config=config,
)

response.content
```

Если вы посмотрите на LangSmith, вы можете увидеть, что именно происходит под капотом в [LangSmith trace](https://smith.langchain.com/public/fa6b00da-bcd8-4c1c-a799-6b32a3d62964/r).

## Потоковая передача

Теперь у нас есть функциональный чат-бот. Однако одним из *действительно* важных UX-соображений для приложений чат-ботов является потоковая передача. LLM иногда могут отвечать долго, поэтому для улучшения пользовательского опыта многие приложения используют потоковую передачу каждого токена по мере его генерации. Это позволяет пользователю видеть ход работы.

На самом деле это очень просто сделать!

Все цепочки предоставляют метод `.stream`, и цепочки, которые используют историю сообщений, не исключение. Мы можем просто использовать этот метод, чтобы получить потоковый ответ.

```python
config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
{
"messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
"language": "English",
},
config=config,
):
print(r.content, end="|")
```

## Дальнейшие шаги

Теперь, когда вы понимаете основы создания чат-бота в LangChain, вот несколько более продвинутых уроков, которые могут вас заинтересовать:

- [Разговорный RAG](/docs/tutorials/qa_chat_history): Реализация чат-бота с использованием внешнего источника данных.
- [Агенты](/docs/tutorials/agents): Создание чат-бота, который может выполнять действия.

Если вы хотите углубиться в детали, вот некоторые полезные материалы:

- [Потоковая передача](/docs/how_to/streaming): потоковая передача *необходима* для приложений чата.
- [Как добавить историю сообщений](/docs/how_to/message_history): более подробная информация о работе с историей сообщений.
```