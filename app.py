import os
import time
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db_arctic"

pipeline_options = PdfPipelineOptions()
pipeline_options.do_table_structure = True
pipeline_options.do_ocr = True

# Initialize the Converter
converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def process_with_docling(file_path):
    print(f"    Analyzing layout...")

    # Initialize converter inside or ensure it's global
    # If converter isn't defined globally, uncomment the next line:
    # converter = DocumentConverter()

    # 1. Convert PDF to structured Markdown
    result = converter.convert(file_path)
    markdown_output = result.document.export_to_markdown()

    # 2. Convert to LangChain format
    docs = [Document(page_content=markdown_output, metadata={"source": file_path})]

    # 3. Tightened Splitter for Snowflake Arctic (Limit: 512 Tokens)
    # We use chunk_size=400 characters to stay safely under the token limit
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=[
             "\n# ",   # Big Chapters
             "\n## ",  # Sections
             "\n### ", # Sub-sections
             "\n\n",   # Paragraphs
             "\n",     # Lines
             ". ",     # Sentence ends (added for better splitting)
             " "       # Words
        ]
    )

    # This splits the documents into safe, Arctic-friendly pieces
    return splitter.split_documents(docs)

def main():
    start_total = time.time()

    # --- STAGE 1: INITIALIZATION ---

    print("Initializing Vector Store & Enbeddings")
    init_start = time.time()
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    print(f"    Initialization took: {time.time() - init_start:.2f}s")

    # --- STAGE 2: SCANNING ---

    existing_items = vectorstore.get(include=[])
    existing_ids = set(existing_items["ids"])
    pdf_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found")
        return

    all_new_chunks = []

    # --- STAGE 3: DOCLING PROCESSING ---

    print(f"\n📂 Found {len(pdf_files)} PDFs. Starting Docling analysis...")

    for file_path in pdf_files:
        file_start = time.time()
        print(f" Processing: {file_path}")

        # This is where the 4-column layout is reconstructed
        file_chunks = process_with_docling(file_path)

        new_in_file = 0
        for i, chunk in enumerate(file_chunks):
            chunk_id = f"{file_path}:arctic:{i}"
            if chunk_id not in existing_ids:
                chunk.page_content = f"passage: {chunk.page_content}"
                chunk.metadata["id"] = chunk_id
                all_new_chunks.append(chunk)
                new_in_file += 1

        file_elapsed = time.time() - file_start
        print(f"   ∟ Done. Created {new_in_file} chunks in {file_elapsed:.2f}s")

    # --- STAGE 4: CHROMA DB PUSH ---
    if all_new_chunks:
        print(f"\n Pushing {len(all_new_chunks)} chunks to ChromaDB (Batching by 100)...")
        push_start = time.time()

        new_chunk_ids = [chunk.metadata["id"] for chunk in all_new_chunks]
        for i in range(0, len(all_new_chunks), 100):
            batch = all_new_chunks[i:i+100]
            batch_ids = new_chunk_ids[i:i+100]
            vectorstore.add_documents(batch, ids=batch_ids)
            print(f"   ∟ Processed {i + len(batch)} / {len(all_new_chunks)}...")

        push_elapsed = time.time() - push_start
        print(f"⏱️  ChromaDB Update took: {push_elapsed:.2f}s")
    else:
        print("\n✅ No new content to add.")

    total_elapsed = time.time() - start_total
    print(f"\nFINISHED. Total Process Time: {total_elapsed / 60:.2f} minutes")

if __name__ == "__main__":
    main()