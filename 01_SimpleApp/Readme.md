## Создание простого приложения с использованием больших языковых моделей (LLM)

В этом кратком руководстве мы покажем вам, как создать простое приложение с использованием больших языковых моделей (LLM). Это приложение будет переводить текст с английского языка на другой язык. Это довольно простое приложение LLM - оно включает всего один вызов LLM и некоторые подсказки. Тем не менее, это отличный способ начать работу с LangChain - множество функций можно реализовать с помощью простых подсказок и вызова LLM!

## Концепции

Мы рассмотрим следующие концепции:

- Использование [языковых моделей](/docs/concepts/#chat-models)
- Использование [шаблонов подсказок](/docs/concepts/#prompt-templates) и [анализаторов выходных данных](/docs/concepts/#output-parsers)
- [Цепочечное связывание](/docs/concepts/#langchain-expression-language) шаблона подсказки + LLM + анализатора выходных данных с помощью LangChain
- Отладка и трассировка вашего приложения с помощью [LangSmith](/docs/concepts/#langsmith)
- Развертывание вашего приложения с помощью [LangServe](/docs/concepts/#langserve)

Довольно много всего! Давайте начнем.

## Настройка

### Jupyter Notebook

Это руководство (и большинство других руководств в документации) использует [Jupyter notebooks](https://jupyter.org/) и предполагает, что вы тоже используете их. Jupyter notebooks отлично подходят для изучения работы с системами LLM, потому что часто возникают проблемы (неожиданный вывод, отказ API и т.д.), а прохождение руководств в интерактивной среде позволяет лучше понять их.

Это и другие учебные материалы лучше всего запускать в Jupyter notebook. Инструкции по установке можно найти [здесь](https://jupyter.org/install).

### Установка

Чтобы установить LangChain, выполните следующую команду:

```{=mdx}
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import CodeBlock from "@theme/CodeBlock";

<Tabs>
<TabItem value="pip" label="Pip" default>
<CodeBlock language="bash">pip install langchain</CodeBlock>
</TabItem>
<TabItem value="conda" label="Conda">
<CodeBlock language="bash">conda install langchain -c conda-forge</CodeBlock>
</TabItem>
</Tabs>

```

Дополнительную информацию можно найти в нашем [Руководстве по установке](/docs/how_to/installation).

### LangSmith

Многие приложения, которые вы будете создавать с помощью LangChain, будут состоять из нескольких шагов с несколькими вызовами LLM.

По мере того как эти приложения становятся все более сложными, крайне важно иметь возможность проверить, что именно происходит внутри вашей цепочки или агента.

Лучший способ сделать это - с помощью [LangSmith](https://smith.langchain.com).

После регистрации по указанной выше ссылке убедитесь, что вы установили переменные среды для начала ведения журнала трассировки:

```shell
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```

Или, если вы работаете в блокноте, вы можете установить их с помощью:

```python
import getpass
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

## Пошаговое руководство

В этом руководстве мы создадим приложение для перевода пользовательского ввода с одного языка на другой.

## Использование языковых моделей

Сначала давайте научимся использовать языковую модель сама по себе. LangChain поддерживает множество различных языковых моделей, которые можно использовать взаимозаменяемо - выберите ту, которую хотите использовать ниже!

```{=mdx}
import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs openaiParams={`model="gpt-4"`} />
```

```python
# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
```

Давайте сначала используем модель напрямую. `ChatModel` - это экземпляры LangChain "выполняемых", что означает, что они предоставляют стандартный интерфейс для взаимодействия с ними. Чтобы просто вызвать модель, можно передать список сообщений в метод `.invoke`.

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
SystemMessage(content="Translate the following from English into Italian"),
HumanMessage(content="hi!"),
]

model.invoke(messages)
```

Если мы включили LangSmith, мы увидим, что этот запуск зарегистрирован в LangSmith, и можем увидеть [трассировку LangSmith](https://smith.langchain.com/public/88baa0b2-7c1a-4d09-ba30-a47985dde2ea/r).

## Анализаторы выходных данных

Обратите внимание, что ответ модели - это `AIMessage`. Он содержит текстовый ответ, а также другие метаданные об ответе. Часто нам может просто понадобиться работать с текстовым ответом. Мы можем извлечь только этот ответ, используя простой анализатор выходных данных.

Сначала импортируем простой анализатор выходных данных.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
```

Один из способов его использования - использовать его самостоятельно. Например, можно сохранить результат вызова языковой модели и затем передать его анализирующему устройству.

```python
result = model.invoke(messages)
```

```python
parser.invoke(result)
```

Чаще всего мы можем "связать" модель с этим анализатором выходных данных. Это означает, что этот анализатор выходных данных будет вызываться каждый раз в этой цепочке. Эта цепочка принимает тип ввода языковой модели (строка или список сообщений) и возвращает тип вывода анализатора выходных данных (строка).

Мы можем легко создать цепочку с помощью оператора `|`. Оператор `|` используется в LangChain для объединения двух элементов.

```python
chain = model | parser
```

```python
chain.invoke(messages)
```

Если мы теперь посмотрим на LangSmith, то увидим, что цепочка состоит из двух шагов: сначала вызывается языковая модель, а затем результат передается анализатору выходных данных. Мы можем увидеть [трассировку LangSmith]( https://smith.langchain.com/public/f1bdf656-2739-42f7-ac7f-0f1dd712322f/r).

## Шаблоны подсказок

Прямо сейчас мы передаем список сообщений непосредственно в языковую модель. Откуда берется этот список сообщений? Обычно он строится на основе комбинации пользовательского ввода и логики приложения. Эта логика приложения обычно берет необработанный пользовательский ввод и преобразует его в список сообщений, готовых для передачи в языковую модель. Распространенные преобразования включают добавление системного сообщения или форматирование шаблона с помощью пользовательского ввода.

Шаблоны подсказок - это концепция в LangChain, разработанная для помощи в этом преобразовании. Они принимают необработанный пользовательский ввод и возвращают данные (подсказку), которые готовы к передаче в языковую модель.

Давайте создадим здесь шаблон подсказки. Он будет принимать два переменных пользователя:

- `language`: язык, на который нужно перевести текст
- `text`: текст для перевода

```python
from langchain_core.prompts import ChatPromptTemplate
```

Сначала создадим строку, которую мы будем форматировать как системное сообщение:

```python
system_template = "Translate the following into {language}:"
```

Затем мы можем создать шаблон подсказки. Это будет комбинация `system_template`, а также простого шаблона для того, где нужно поместить текст.

```python
prompt_template = ChatPromptTemplate.from_messages(
[("system", system_template), ("user", "{text}")]
)
```

Ввод для этого шаблона подсказки - это словарь. Мы можем поиграть с этим шаблоном подсказки самостоятельно, чтобы увидеть, что он делает.

```python
result = prompt_template.invoke({"language": "italian", "text": "hi"})

result
```

Мы видим, что он возвращает `ChatPromptValue`, который состоит из двух сообщений. Если мы хотим получить доступ к сообщениям напрямую, то сделаем следующее:

```python
result.to_messages()
```

Теперь мы можем объединить это с моделью и анализатором выходных данных, которые были выше. Это объединит все три компонента.

```python
chain = prompt_template | model | parser
```

```python
chain.invoke({"language": "italian", "text": "hi"})
```

Если мы посмотрим на трассировку LangSmith, мы увидим, что все три компонента отображаются в [трассировке LangSmith](https://smith.langchain.com/public/bc49bec0-6b13-4726-967f-dbd3448b786d/r).

## Обслуживание с помощью LangServe

Теперь, когда мы создали приложение, нужно его запустить. Вот где появляется LangServe.
LangServe помогает разработчикам развертывать цепочки LangChain в качестве REST API. Вам не нужно использовать LangServe для использования LangChain, но в этом руководстве мы покажем, как развернуть ваше приложение с помощью LangServe.

В то время как первая часть этого руководства была предназначена для запуска в Jupyter Notebook или скрипте, теперь мы выйдем за его пределы. Мы создадим файл Python, а затем будем взаимодействовать с ним из командной строки.

Установите с помощью:

```bash
pip install "langserve[all]"
```

### Сервер

Чтобы создать сервер для нашего приложения, мы создадим файл `serve.py`. Он будет содержать нашу логику для запуска нашего приложения. Он состоит из трех частей:

1. Определение цепочки, которую мы только что построили выше.
2. Наше приложение FastAPI
3. Определение маршрута, с которого будет запускаться цепочка, что делается с помощью `langserve.add_routes`

```python
#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
[("system", system_template), ("user", "{text}")]
)

# 2. Create model
model = ChatOpenAI()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser


# 4. App definition
app = FastAPI(
title="LangChain Server",
version="1.0",
description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
app,
chain,
path="/chain",
)

if __name__ == "__main__":
import uvicorn

uvicorn.run(app, host="localhost", port=8000)
```

И все! Если мы выполним этот файл:

```bash
python serve.py
```

мы должны увидеть, что наша цепочка запускается по адресу [http://localhost:8000](http://localhost:8000).

### Площадка

Каждая служба LangServe поставляется с простым [встроенным пользовательским интерфейсом](https://github.com/langchain-ai/langserve/blob/main/README.md#playground) для настройки и вызова приложения с потоковым выводом и видимостью промежуточных шагов.
Перейдите по адресу [http://localhost:8000/chain/playground/](http://localhost:8000/chain/playground/), чтобы попробовать! Передайте те же входные данные, что и раньше - `{"language": "italian", "text": "hi"}` - и он должен ответить так же, как и раньше.

### Клиент

Теперь давайте настроим клиент для программного взаимодействия с нашей службой. Мы можем легко сделать это с помощью `[langserve.RemoteRunnable](/docs/langserve/#client)`.
С помощью этого мы можем взаимодействовать с запускаемой цепочкой так, как если бы она запускалась на стороне клиента.

```python
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
remote_chain.invoke({"language": "italian", "text": "hi"})
```

Чтобы узнать больше о многих других функциях LangServe, [перейдите сюда](/docs/langserve).

## Заключение

Вот и все! В этом руководстве мы рассмотрели создание нашего первого простого приложения LLM. Мы узнали, как работать с языковыми моделями, как анализировать их выходные данные, как создавать шаблон подсказки, как получить отличную наблюдаемость в цепочках, которые вы создаете с помощью LangSmith, и как развертывать их с помощью LangServe.

Это лишь поверхностное знакомство с тем, что вам нужно будет изучить, чтобы стать опытным инженером по искусственному интеллекту. К счастью, у нас есть множество других ресурсов!

Дополнительные подробные учебные материалы можно найти в разделе [Учебные материалы](/docs/tutorials).

Если у вас есть конкретные вопросы о том, как выполнить определенные задачи, обратитесь к разделу [Руководства](/docs/how_to).

Чтобы прочитать о ключевых концепциях LangChain, мы подготовили подробные [Концептуальные руководства](/docs/concepts).