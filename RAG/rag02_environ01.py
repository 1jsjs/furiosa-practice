#rag01 copy
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0, 
    # openai_api_key = openai_api_key
)

response = llm.invoke ('너는 누구니')

print (response.content)