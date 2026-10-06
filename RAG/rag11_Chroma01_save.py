
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


#01. 데이터 불러오기
loader1 = TextLoader(data_path + 'samsung_outlook.txt', encoding='utf-8')
loader2 = TextLoader(data_path + 'nvidia_outlook.txt', encoding='utf-8')
# loader3= TextLoader(data_path + '2026_AI_for_All.txt')


#02. 문서를 자른다 (청킹)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=['\n\n', '\n', ' ', ''], #통상 default
)
split_doc1 = loader1.load_and_split(text_splitter) # 청크 300, 오버랩 100
split_doc2 = loader2.load_and_split(text_splitter) # 청크 300, 오버랩 100

# print (split_doc1)
# print (len(split_doc1), len(split_doc2)) #9 9

embeddings = OpenAIEmbeddings (
    model = "text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5, #차원 디멘션 강제
)

DB_PATH = "./_db/Chroma11/"

#저장
db = Chroma.from_documents (
    documents= split_doc1 + split_doc2,
    embedding= embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma11",
)

print ("Chroma 문서 저장 끝")
