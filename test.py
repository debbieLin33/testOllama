from langchain_community.llms import Ollama
llm = Ollama(model="llama2:7b-chat")
ans = llm.invoke("Tell me a joke")
print(ans)

