import os
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.getenv("OPENAI_API_MODEL", "gpt-5")
llm = ChatOpenAI(model=MODEL, temperature=0)
with open("examples.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
example_prompt = PromptTemplate.from_template("Вопрос: {question}\nОтвет: {answer}")
examples = [value for value in data['code_examples'].values()]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Ты опытный помошник по программированию на python:",
    suffix="Ввод: {input}\nВывод:",
    input_variables=["input"]
)

formatted_prompt = prompt.format(input="Как найти ключ по значению в словаре?")
response = llm.invoke(formatted_prompt)

print(response.content)