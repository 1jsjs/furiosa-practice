# rag 8-2 copy
#LCEL : LangChain Expression Languade
#pmo : prompt - model - output
#plp : prompt - llm - parser
# chain  : prompt | model | output_parser
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# chain-prompt
prompt = "삼성전자의 창업주는 누구인가요?"

embeddings = OpenAIEmbeddings (
    model = "text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)

vector =embeddings.embed_query (prompt)
print (vector)
print ("===============================")
print ("임베딩 벡터의 차원 :", len(vector))
"""
===============================
임베딩 벡터의 차원 : 1536
"""