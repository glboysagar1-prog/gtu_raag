import json
import pickle
import chromadb

def verify():
    # 1. Verify JSON chunks
    print("Loading chunks...")
    with open("gtu_chunks/all_chunks_perplexity.json", "r") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")
    if chunks:
        print(f"Sample chunk keys: {chunks[0].keys()}")

    # 2. Verify BM25
    print("\nLoading BM25 index...")
    with open("gtu_bm25_index/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    print(f"BM25 index loaded successfully. Type: {type(bm25)}")

    # 3. Verify Chroma
    print("\nLoading Chroma Vector Store...")
    client = chromadb.PersistentClient(path="gtu_vector_index")
    # list collections
    collections = client.list_collections()
    if collections:
        collection_names = [c.name for c in collections]
        print(f"Found Chroma Collections: {collection_names}")
        coll = client.get_collection(collection_names[0])
        print(f"Collection {collection_names[0]} has {coll.count()} items.")
    else:
        print("No collections found in Chroma DB!")

if __name__ == "__main__":
    verify()
