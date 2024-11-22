from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, AIMessage
# use the Ollama model to generate the response
llm = Ollama(model="llama3.1")

# create a prompt template
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])
# load the documents
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
# split the documents
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# create the embeddings
embeddings = OllamaEmbeddings()
# create the vector store
vector = FAISS.from_documents(documents, embeddings)

# create the retriever: 多輪對話
retriever = vector.as_retriever()
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
result = retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})

print("Model Output:", result)