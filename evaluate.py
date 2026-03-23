import os
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
# Assuming your app.py has the initialize_rag function
from app import initialize_rag

# --- STEP 1: SETUP THE LOCAL JUDGE ---
# We use a larger model (12B) for judging to ensure high-quality reasoning
local_judge = OllamaModel(
    model="gemma3:12b",
    base_url="http://localhost:11434",
    temperature=0
)

# --- STEP 2: SETUP YOUR RAG SYSTEM ---
rag_chain = initialize_rag("gemma3:4b", k=5, ctx=8192)

# --- STEP 3: RUN THE ARCHIVE QUERY ---
query = "What was discussed regarding the 1994 VVA budget?"
prefixed_query = f"search_query: {query}"

source_documents = retriever.invoke(prefixed_query)
context_chunks = [doc.page_content for doc in source_documents]

# Get the final AI answer
actual_response = rag_chain.invoke(prefixed_query)

metric = FaithfulnessMetric(
    threshold=0.7,
    model=local_judge,
    include_reason=True
)

test_case = LLMTestCase(
    input=query,
    actual_output=actual_response,
    retrieval_context=context_chunks
)

# --- STEP 5: MEASURE & REPORT ---
metric.measure(test_case)

print("\n" + "="*45)
print(f"📊 EVALUATION REPORT")
print("="*45)
print(f"Faithfulness Score: {metric.score:.2f}")
print(f"Pass Threshold:     {metric.threshold}")
print(f"Score: {metric.score:.2f} | Status: {'✅ PASS' if metric.is_successful() else '❌ FAIL'}")
print(f"Reasoning:\n{metric.reason}")
print("="*45 + "\n")