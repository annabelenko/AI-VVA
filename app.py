import streamlit as st
import os
import psutil
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

# 1. Page Configuration & Tailwind-style Styling
st.set_page_config(page_title="VVA Archive Explorer", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stTextInput > div > div > input { border-radius: 0.5rem; border: 2px solid #e2e8f0; }
    .stats-card { background-color: #000000; padding: 1rem; border-radius: 0.75rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar for Hardware & AI Stats
with st.sidebar:
    st.title("AI Token Stats")
    stats_placeholder = st.empty()
    st.divider()

    st.title("AI Control Panel")
    #Model Selection - From 270M to 27b

    selected_model = st.selectbox(
       "Choose AI Brain",
       ["gemma3:270m", "gemma3:1b","gemma3:4b", "gemma3:12b", "gemma3:27b"],
       index=1, #Default to 1b
       help="27b is the most intelligent but uses the most RAM and CPU power."
    )

    #Retrieval K-Value Slider
    k_val = st.slider(
       "Documents to Analyze (K)",
       min_value=1, max_value=20, value=7,
       help="How many chunks of the archive should the AI read before answering?"
    )

    st.divider()
    if st.button("Clear App Cache"):
       st.cache_resource.clear()
       st.success("Cache cleared!")

# 3. Backend Logic (Cached to avoid reloading the 12B model constantly)
@st.cache_resource

def initialize_rag(model_name, k):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOllama(model=model_name)
    
    # Simplified RAG template
    template = """
    Answer the question below using only this information:
    {context}

    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # This chain handles retrieval and generation
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

rag_chain = initialize_rag(selected_model, k_val)

# 4. The User Interface
st.title("🪖 VVA Veteran Archive Explorer")
st.write("Currently powered by **{selected_model}** reading **{k_val} chunks** per query.")

query = st.text_input("Enter your research question:", placeholder="e.g., What was discussed regarding homeless veterans?")

if query:
    with st.spinner("Analyzing archives..."):
        # Wrap the execution in the callback to capture 'Stats'
        with get_openai_callback() as cb:
            response = rag_chain.invoke(query)
            
            # Update the sidebar stats
            with stats_placeholder.container():
                st.write(f"**Total Tokens:** {cb.total_tokens}")
                st.write(f"**Prompt Tokens:** {cb.prompt_tokens}")
                st.write(f"**Completion Tokens:** {cb.completion_tokens}")
                st.info("Note: Ollama provides token counts via the OpenAI callback compatibility layer.")

        # Display the result in a clean card
        st.markdown("### 🤖 AI Analysis")
        st.markdown(f'<div class="stats-card">{response}</div>', unsafe_allow_html=True)

        # Optional: Button to clear cache if you update the PDF
        if st.button("Clear Cache"):
            st.cache_resource.clear()
