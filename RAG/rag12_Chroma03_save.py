#11-1 copy
import os

from glob import glob #폴더에서 텍스트 파일 목록 가져오기

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"
data_path = "./_data/rag_data/"
DB_PATH = "./_db/Chroma12/"

txt_files = glob (os.path.join(data_path, "*.txt"))
# print (txt_files)
# ['./_data/rag_data\\2026_AI_for_All.txt', './_data/rag_data\\nvidia_outlook.txt', './_data/rag_data\\samsung_outlook.txt']


#01. 데이터 불러오기
data = []
for text_file in txt_files:
    loader = TextLoader (text_file, encoding="utf-8")
    data += loader.load()
# print ("==========================================")
# print (data[0])
# print ("==========================================")
# print (len(data)) #3
# print (data[0].page_content) #우리가 알고 있는 데이터의 문장들

char_count = [len(doc.page_content) for doc in data]
# print (char_count) #[8158, 2049, 1898]


#02. 문서를 자른다 (청킹)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=10,
    separators=['\n\n', '\n', ' ', ''], #통상 default
)
texts = text_splitter.split_documents (data)
# print ("생성된 텍스트 청크 수 : ", len(texts)) #ㅠ생성된 텍스트 청크 수 :  52
# print ("각 청크의 길이 : ", list(len(text.page_content) for text in texts))
"""각 청크의 길이 :  [259, 282, 282, 128, 276, 214, 158, 249, 262, 291, 268, 182, 286, 295, 182, 162, 283, 286,
257, 235, 214, 258, 207, 286, 220, 198, 271, 57, 272, 122, 9, 269, 299, 284, 289, 209, 222, 230, 254, 249,
296, 90, 247, 243, 185, 219, 239, 235, 298, 282, 187, 249]"""
# print ("첫번째 청크의 내용 : ", texts[0].page_content)
# print ("첫번째 청크의 길이 : ", len(texts[0].page_content))
# print ("두번째 청크의 내용 : ", len(texts[1].page_content))


#3. 임베딩
embeddings = OpenAIEmbeddings (
    model = "text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    # dimensions=5, #차원 디멘션 강제
)
sample_text = "삼성전자의 창업자는 누구인가요?"
vector = embeddings.embed_query(sample_text)
# print (len(vector)) #1536


#4. 저장
vector_store = Chroma.from_documents (
    documents= texts,
    embedding= embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma12",
)
print (f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}",) #벡터 저장소에 저장된 문서 수 : 52

query = "삼성전자의 창업자는 누구인가요?"
result = vector_store.similarity_search(query)
print (f"검색 결과의 길이 : {len(result)}") #검색 결과의 길이 : 4

##################################### Retrievers ##########################################
##################################### 검색기 ##########################################
retriever =  vector_store.as_retriever(search_kwargs={"k":2})

print (retriever) #tags=['Chroma', 'OpenAIEmbeddings'] vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x0000026429AF7A10> search_kwargs={'k': 2}
aaa = retriever.invoke(query)
print (f"검색된 관련 문서 수 : {len(aaa)}") #검색된 관련 문서 수 : 2
print (f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}")
"""첫번째 관련 문서 내용 미리보기 : 삼성전자 사업 전망

삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사"""
