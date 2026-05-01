import matplotlib
matplotlib.use('Agg')  # Required for headless Linux servers
import matplotlib.pyplot as plt
import umap
import chromadb
import numpy as np
import os

def get_data(folder_path):
    """
    Connects to a specific folder, finds the internal collection,
    and extracts embeddings + metadata.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # 1. Initialize client for this specific database folder
    client = chromadb.PersistentClient(path=folder_path)

    # 2. Find the collection name (LangChain usually defaults to 'langchain')
    collections = client.list_collections()
    if not collections:
        raise ValueError(f"No collections found in {folder_path}")

    col_name = collections[0].name
    col = client.get_collection(col_name)
    count = col.count()
    print(f"Folder {folder_path} contains {count} documents.")

    if count == 0:
        raise ValueError(f"The collection '{col_name}' in {folder_path} is empty.")

    print(f"Fetching IDs for {col_name}...")
    all_ids = col.get()['ids']

    # 3. Pull the data
    print(f"Requesting embeddings for {len(all_ids)} vectors ...")
    data = col.get(ids=all_ids, include=['embeddings', 'metadatas'])

    if data['embeddings'] is None or len(data['embeddings']) == 0:
        raise ValueError(f"No embeddings found in {folder_path}. Is the DB empty?")

    return np.array(data['embeddings']), data['metadatas']

# --- MAIN EXECUTION ---

# 1. Extract Data from your specific folders
print("Starting data extraction...")
try:
    nomic_vecs, nomic_meta = get_data("chroma_db_nomic_prefixed")
    arctic_vecs, arctic_meta = get_data("chroma_db_arctic")
except Exception as e:
    print(f"❌ Error during extraction: {e}")
    exit(1)

# 2. Configure UMAP
# n_neighbors: Low values (5-15) focus on local clusters (good for topical clustering)
# min_dist: Low values (0.1) allow for tighter groups
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)

# 3. Generate 2D Projections
print(f"Projecting Nomic Manifold ({nomic_vecs.shape})...")
nomic_2d = reducer.fit_transform(nomic_vecs)

print(f"Projecting Arctic Manifold ({arctic_vecs.shape})...")
arctic_2d = reducer.fit_transform(arctic_vecs)

# 4. Create Side-by-Side Graph
print("Generating plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot Nomic
ax1.scatter(nomic_2d[:, 0], nomic_2d[:, 1], s=8, alpha=0.5, c='#3498db', edgecolors='none')
ax1.set_title("Nomic-Embed-v1.5 (768d, Prefixed)", fontsize=14)
ax1.set_xlabel("UMAP Dimension 1")
ax1.set_ylabel("UMAP Dimension 2")

# Plot Arctic
ax2.scatter(arctic_2d[:, 0], arctic_2d[:, 1], s=8, alpha=0.5, c='#e74c3c', edgecolors='none')
ax2.set_title("Snowflake Arctic (1024d, Raw)", fontsize=14)
ax2.set_xlabel("UMAP Dimension 1")
ax2.set_ylabel("UMAP Dimension 2")

plt.suptitle("VVA Archive Analysis: Geometric Topical Clustering", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the file
output_file = "embedding_comparison.png"
plt.savefig(output_file, dpi=300)
print(f"🚀 Success! Graph saved as {output_file}")