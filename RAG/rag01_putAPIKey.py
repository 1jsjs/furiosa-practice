from langchain_openai import ChatOpenAI

openai_api_key = ""

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0, 
    openai_api_key = openai_api_key
)

response = llm.invoke ('s나는 귀구나')

print (response.content)