# rag 20-1 copy

#LCEL : LangChain Expression Languade
#pmo : prompt - model - output
#plp : prompt - llm - parser
# chain  : prompt | model | output_parser
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# chain-prompt
prompt = "삼성전자의 창업주는 누구인가요?"

# pip install HuggingFaceEmbeddings
# pip install sentence-transformers
embeddings = HuggingFaceEmbeddings (
    model_name = "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs = {
        "device" : "cpu",
        # "local_files_only" : True,
    }
)

vector =embeddings.embed_query (prompt)
print (vector)
print ("===============================")
print ("임베딩 벡터의 차원 :", len(vector)) # 임베딩 벡터의 차원 : 1024