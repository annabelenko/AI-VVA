from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Connect to the 5.1MB database you just built
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text")
)

# 2. Setup the "Retriever" (This pulls the top 5 most relevant chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Define the LLM (Using your 12B Gemma model)
llm = ChatOllama(model="gemma3:12b")

# 4. The Prompt: This tells the AI to ONLY use the PDF data
template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. Create the "Chain" (The LangChain Logic)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Test it out!
question = "What are the main topics discussed in this 1999 veteran document?"
print(f"\nQuestion: {question}")
print("\nAI Response:")
print(chain.invoke(question))
