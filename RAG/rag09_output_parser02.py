# 8-1 copy
#LCEL : LangChain Expression Languade
#pmo : prompt - model - output
#plp : prompt - llm - parser
# chain  : prompt | model | output_parser

template = """
당신은 영어를 가르치는 10년차 영어 선냉님입니다.
주어진 상황에 맞는 영어 회화에 맞는 영어회화를 작성해 주세요.
양식은  [FORMAT]을 참고하여 작성해주세요.

#상황:
{question}

#FORMAT:
-영어회화 :
-한글번역 :
"""
import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# chain-prompt
prompt = PromptTemplate.from_template(template=template)

#chain - model
model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

#chain - outputparser
output_parser = StrOutputParser()
chain = prompt | model | output_parser #chain으로 엮기

input = {"question" : "나는 집에 가고 싶다."}

response = chain.invoke(input)

print (response)