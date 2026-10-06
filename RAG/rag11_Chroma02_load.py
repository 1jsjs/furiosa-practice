#11-1 copy
import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"
data_path = "./_data/rag_data/"
DB_PATH = "./_db/Chroma11/"

embeddings = OpenAIEmbeddings (
    model = "text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5, #차원 디멘션 강제
)

# 불러오기
db = Chroma (
    embedding_function= embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma11",
)

# 저장된 데이터 확인
print ("==========================================")
print (db.get())
print ("Chroma 문서 불러오기 끝")

print ("==========================================")
aaa = db.similarity_search("삼성전자 사업전망에 대해 알려줘.", k=2) # 디폴트는 4
print (aaa)