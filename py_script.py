import os

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

city = 'Самара'
load_dotenv()
MODEL = os.getenv("OPENAI_API_MODEL", "gpt-5")


class WeatherInfo(BaseModel):
    city: str = Field(description="Название города")
    temperature: float = Field(description="Температура в градусах Цельсия")
    condition: str = Field(description="Погодные условия")


py_parser = PydanticOutputParser(pydantic_object=WeatherInfo)

prompt = PromptTemplate(template=
                        """Верни информацию о погоде в городе {city}

                        {format_instructions}"""
                        , input_variables=["city"],
                        partial_variables={"format_instructions": py_parser.get_format_instructions()})

llm = ChatOpenAI(model=MODEL, temperature=0.5)
llm_chain = prompt | llm | py_parser

try:
    result = llm_chain.invoke(
        {'city': city})
    print(result.model_dump_json())
except Exception as e:
    print({'error': str(e)})

