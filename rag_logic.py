import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- HELPER FUNCTIONS ---

def format_docs(docs):
    """
    Combines retrieved Document objects into a single string.
    Ensures the LLM receives clean text context rather than raw Python objects.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# --- CORE INITIALIZATION ---

def initialize_rag(model_name="gemma3:4b",
                   db_path="./chroma_db_nomic_prefixed",
                   embed_model_name="nomic-embed-text",
                   k=5,
                   ctx=8192):
    """
    Initializes a complete RAG chain and its corresponding retriever.

    Args:
        model_name (str): The Ollama LLM to use for generation (e.g., gemma3:4b).
        db_path (str): Folder path for the specific ChromaDB (Nomic vs Arctic).
        embed_model_name (str): The Ollama embedding model (must match ingestion model).
        k (int): Number of document chunks to retrieve per query.
        ctx (int): Context window size for the LLM.

    Returns:
        tuple: (rag_chain, retriever)
    """

    # 1. Initialize Embeddings
    # This serves as the 'Encoder' half of your Sovereign AI setup.
    embeddings = OllamaEmbeddings(model=embed_model_name)

    # 2. Load the Vector Store
    # This points to your 1,831-chunk VVA archive.
    if not os.path.exists(db_path):
        print(f"⚠️ Warning: Database path '{db_path}' not found.")

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # 3. Configure the Retriever
    # Pulls the top 'k' most relevant clusters from your latent manifold.
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 4. Initialize the Local LLM
    # Optimized for your dual 8-core CPUs and 121GiB RAM.
    llm = ChatOllama(
        model=model_name,
        num_ctx=ctx,
        temperature=0  # Zero temperature for reproducible scientific evaluation
    )

    # 5. Define the Research-Grade Prompt
    # Specifically styled for your Vietnam Veterans Archive project.
    template = """
    You are an expert research assistant for the Vietnam Veterans Archive.
    Use the following pieces of retrieved context to answer the user's question.
    If the context does not contain the information needed to answer,
    honestly state that the information is not available in the archive.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 6. Construct the RAG Chain
    # Flow: Context Retrieval -> Text Formatting -> Prompt Filling -> LLM -> String Output
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# --- QUICK TEST BLOCK ---
if __name__ == "__main__":
    # This block only runs if you execute 'python rag_logic.py' directly.
    # It allows you to verify the server can see the database.
    try:
        test_chain, test_retriever = initialize_rag()
        chunk_count = test_retriever.vectorstore._collection.count()
        print(f"✅ Success! RAG logic ready with {chunk_count} chunks.")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")