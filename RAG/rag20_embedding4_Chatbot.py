import os
import gradio as gr

from dotenv import load_dotenv

from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

DB_PATH = "./_db/Chroma13/"
pdf_path = "./_data/"
pdf_loader = PyPDFLoader(pdf_path + 'attention is all you need.pdf')
pdf_docs = pdf_loader.load()

#1. PDF 로딩 및 청크 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunks = pdf_loader.load_and_split(splitter)
print(f"문서의 길이: {len(chunks)}")

### 임베딩
embeddings = HuggingFaceEmbeddings (
    model_name = "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs = {
        "device" : "cpu",
        # "local_files_only" : True,
    }
)

###VectorDB에 삽입하기

db = Chroma.from_documents (
    documents= chunks,
    embedding= embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma13",
)

vector_store = Chroma(
    embedding_function = embeddings,
    persist_directory = DB_PATH,
    collection_name = "chroma13",
)

print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}")
# query = "삼성전자의 창업자는 누구인가요?"
# result = vector_store.similarity_search(query)
# print(f"검색 결과의 길이 : {len(result)}")

############################ Retrievers ############################
############################ 검색기 ############################
retriever = vector_store.as_retriever(search_kwargs={"k":5})
print(retriever)
# aaa = retriever.invoke(query)
# print(f"검색된 관련 문서 수 : {len(aaa)}")
# print(f"첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}...")

print("============================================================================")

############################ 모델 연결 ############################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5-nano-chat',
    temperature=0,  # 0 = 있는 그대로, 1 = 창의적으로
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)



prompt = ChatPromptTemplate.from_template("""
검색된 컨텍스트에 근거해 답변하세요.
답변에 근거 페이지를 [p. 페이지번호] 형식으로 표시하세요.
컨텍스트에서 근거를 찾을 수 없으면 논문에서 확인할 수 없다고 답하세요.

컨텍스트: {context}
질문: {input}
답변:
""")
document_prompt = PromptTemplate.from_template("[p.{page}] {page_content}")

# 체인 생성
docu_chain = create_stuff_documents_chain(model, prompt, document_prompt=document_prompt)    # prompt | model (프롬프트와 모델 연결)
rag_chain = create_retrieval_chain(retriever, docu_chain)   # 검색 | docu_chain

query = "Transformer의 모델 아키텍처를 자세히 설명해줘"
docs = vector_store.similarity_search(query, k=4)

for i, doc in enumerate(docs, 1):
    print(f"\n--- 검색 결과 {i}, PDF 페이지 {doc.metadata.get('page')} ---")
    print(doc.page_content)

######################## Gradio 챗봇 ########################
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({"input" : message})
    return response['answer']

# Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title='Chat Bot')

# Gradio 실행
demo.launch(share=True)