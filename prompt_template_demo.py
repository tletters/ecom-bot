"""
Шаблон предназначен для получения карты изучения технологии с параметризированными уровнями освоения.
принимает параметры:
technology - название техноголоии
start_learn_level - начальный уровень освоения
end_learn_level - конечный уровень освоения
"""

import os

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL_NAME = os.getenv("OPENAI_API_MODEL")
conv_1 = {"configurable": {"thread_id": "conversation_001"}}
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

template_str = '''Ты - опытный наставник и учитель по программированию. Твоя задача составить роадмап изучения технологии "{technology}" от {start_learn_level} до {end_learn_level} уровня. Формат ответа: Ответ должен быть представлен в виде последовательного списка из 5-10 шагов.'''
template = PromptTemplate.from_template(template_str)

prompt_strobj_1 = template.invoke(
    {"technology": "docker", "start_learn_level": 'начального', "end_learn_level": 'среднего'})
prompt_strobj_2 = template.invoke(
    {"technology": "web_socker", "start_learn_level": 'среднего', "end_learn_level": 'продвинутого'})
prompt_strobj_3 = template.invoke(
    {"technology": "kafka", "start_learn_level": 'junior', "end_learn_level": 'senior'})

for prompt in [prompt_strobj_1, prompt_strobj_2, prompt_strobj_3]:
    response = llm.invoke(prompt)
    print(response.content, end='\n\n')

