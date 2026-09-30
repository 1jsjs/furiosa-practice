# 08-1 copy
#LCEL : LangChain Expression Languade
#pmo : prompt - model - output
#plp : prompt - llm - parser
# chain  : prompt | model | output_parser
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# chain-prompt
prompt = PromptTemplate.from_template("{topic}에 대해 {how} 설명해주세요")

#chain - model
model = ChatOpenAI(
    model_name='claude-opus-4-7',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

#chain - outputparser
chain = prompt | model

input = {"topic" : "양자 컴퓨팅 학습 원리", "how" : "초등학생도 이해하기 쉽게"}

response = chain.invoke(input)

print (response.content)