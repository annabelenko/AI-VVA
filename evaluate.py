import os
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1800"
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from rag_logic import initialize_rag

# --- CONFIGURATION ---
# The Judge (Gemma 3: 27b is great for your 121GB RAM)
local_judge = OllamaModel(
    model="gemma3:27b",
    base_url="http://localhost:11434",
    temperature=0
)

# The questions you want to benchmark
test_queries = [
    "What was discussed regarding the 2000 VVA budget?",
    "What are the primary eligibility requirements for veteran healthcare?"
]

# The competitors for your "Smarter, Not Larger" study
configs = [
    {
        "name": "Nomic (Optimized/768d)",
        "db": "./chroma_db_nomic_prefixed",
        "embed": "nomic-embed-text",
        "prefix": "search_query: "
    },
    {
        "name": "Arctic (Baseline/1024d)",
        "db": "./chroma_db_arctic",
        "embed": "snowflake-arctic-embed",
        "prefix": ""
    }
]

# --- THE TOURNAMENT LOOP ---
final_scores = {}

for config in configs:
    print(f"\n🚀 STARTING EVALUATION: {config['name']}")

    # Initialize the specific RAG system
    rag_chain, retriever = initialize_rag(
        model_name="gemma3:4b",
        db_path=config['db'],
        embed_model_name=config['embed']
    )

    total_score = 0

    for query in test_queries:
        prefixed_query = f"{config['prefix']}{query}"

        print(f"  🔍 Retrieving: {query}")
        source_documents = retriever.invoke(prefixed_query)
        context_chunks = [doc.page_content for doc in source_documents]

        print("  🤖 Generating Answer...")
        actual_response = rag_chain.invoke(prefixed_query)

        # Audit the answer
        metric = FaithfulnessMetric(threshold=0.7, model=local_judge, include_reason=True)
        test_case = LLMTestCase(input=query, actual_output=actual_response, retrieval_context=context_chunks)

        metric.measure(test_case)
        total_score += metric.score
        print(f"  ✅ Faithfulness Score: {metric.score:.2f}")

        # NEW: Manual Debugging Block
        if metric.score < 1.0:
            print("\n--- 🔍 DEBUGGING THE FAILURE ---")
            print("TOP RETRIEVED CHUNK:")
            print(context_chunks[0][:500] + "...") # See the first 500 chars of the top chunk

            print("\nJUDGE'S REASONING:")
            print(metric.reason if metric.reason else "Judge did not provide a specific reason string.")

        # If using newer DeepEval versions, try this:
        if hasattr(metric, 'verbose_logs'):
           print(f"VERBOSE LOGS: {metric.verbose_logs}")

        # Calculate average for this model
        final_scores[config['name']] = total_score / len(test_queries)

# --- FINAL POSTER REPORT ---
print("\n" + "="*45)
print(f"📊 FINAL POSTER RESULTS")
print("="*45)
for name, score in final_scores.items():
    status = "🏆 WINNER" if score == max(final_scores.values()) else "Runner-up"
    print(f"{name:25} | Avg Score: {score:.2f} | {status}")
print("="*45 + "\n")