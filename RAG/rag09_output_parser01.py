# 8-1 copy
#LCEL : LangChain Expression Languade
#pmo : prompt - model - output
#plp : prompt - llm - parser
# chain  : prompt | model | output_parser
import os

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# chain-prompt
prompt = PromptTemplate.from_template("{topic}에 대해 설명해주세요")

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

input = {"topic" : "양자 컴퓨팅 학습 원리"}

response = chain.invoke(input)

print (response)