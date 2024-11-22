# serving application
from typing import List
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langserve import add_routes
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
# 1. Load Retriever

# loader = WebBaseLoader("https://www.kgilife.com.tw/zh-tw/about-us/about-us")
loader = PyPDFLoader("sake.pdf")
docs = loader.load()

# Split the documents into paragraphs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
documents = text_splitter.split_documents(docs)
# Embed the documents
# embeddings = OllamaEmbeddings()
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings': False}
)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()


# use the Ollama model to generate the response
llm = Ollama(model="llama3.1")

# create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}"),
    ('system', '請依照以下文章:"{context}"內容回答問題。請按照原文，勿回答文章以外的資訊或竄改文字。'),
])
retriever = vector.as_retriever()
# print(retriever.get_relevant_documents("發酵食品列舉"))
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 4. App definition
app = FastAPI()
class InputModel(BaseModel):
    question: str
@app.post("/ask")
def read_root(request: InputModel):
    response = retrieval_chain.invoke({
        'input': request.question,
    })
    print(response)
    return response
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)