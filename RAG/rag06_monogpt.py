# rag05 copy

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

llm = ChatOpenAI(
    model_name='gpt-5-nano',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

response = llm.invoke ('너는 누구니')

print (response.content)