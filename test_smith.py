from langchain_ollama import ChatOllama
# This simple call should trigger a trace
llm = ChatOllama(model="gemma3:270m")
llm.invoke("Hello LangSmith!")
print("Check your dashboard now!")
