import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"

def main():
    # 1. Load PDFs
    print(f"--- Scanning {DATA_PATH} ---")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("No PDFs found!")
        return

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Create Unique IDs (The "Fingerprint")
    # This creates an ID like "data/12-1999.pdf:page:5:chunk:2"
    chunks_with_ids = []
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        
        # Add the ID to the metadata so we can track it
        chunk.metadata["id"] = chunk_id
        chunks_with_ids.append(chunk)

    # 4. Initialize Vector Store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 5. Filter out existing chunks
    existing_items = vectorstore.get(include=[]) # Get all existing IDs
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing items in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Adding {len(new_chunks)} new chunks to the database...")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vectorstore.add_documents(new_chunks, ids=new_chunk_ids)
        print("✅ Success! New documents integrated.")
    else:
        print("✅ No new documents to add. Database is already up to date.")

if __name__ == "__main__":
    main()
