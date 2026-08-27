import faiss
import os 
import numpy as np
import pickle
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_PATH = "faiss_index.bin"
DOCSTORE_PATH = "docstore.pkl" # <-- New file to hold the text

GLOBAL_INDEX = None
GLOBAL_DOCS = None
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_faiss_index(index_path, docstore_path):
    global GLOBAL_INDEX, GLOBAL_DOCS
    
    # 1. LOAD FROM DISK IF IT EXISTS
    if os.path.exists(index_path) and os.path.exists(docstore_path):
        print(f"Loading existing FAISS index and docstore...")
        GLOBAL_INDEX = faiss.read_index(index_path)
        
        with open(docstore_path, "rb") as f:
            GLOBAL_DOCS = pickle.load(f)
        return

    # 2. INGEST DATA IF NO INDEX EXISTS
    print("Building new index from scratch...")
    if os.path.exists("data.txt"):
        documents = open("data.txt", "r").read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
            keep_separator=False
        )
        # split_text returns a list of strings
        GLOBAL_DOCS = text_splitter.split_text(documents)
    
    # Embed and cast to float32
    vectors = embeddings.embed_documents(GLOBAL_DOCS)
    vectors = np.array(vectors, dtype=np.float32)
    dim = vectors.shape[1]
    
    # Build FAISS index
    GLOBAL_INDEX = faiss.IndexFlatL2(dim)
    GLOBAL_INDEX.add(vectors)
    
    # 3. SAVE TO DISK
    print("Saving FAISS index and docstore...")
    faiss.write_index(GLOBAL_INDEX, index_path)
    
    with open(docstore_path, "wb") as f:
        pickle.dump(GLOBAL_DOCS, f)

def search_faiss(query: str, k: int = 2):
    # 1. Embed the search query
    query_vector = embeddings.embed_query(query)
    
    # 2. FAISS expects a 2D array, so we wrap it in a list
    query_vector = np.array([query_vector], dtype=np.float32)
    
    # 3. Search! Returns distances and the IDs of the matched vectors
    distances, indices = GLOBAL_INDEX.search(query_vector, k)
    
    # 4. Map the IDs back to the actual text
    results = []
    for idx in indices[0]: # indices is a 2D array, grab the first row
        if idx != -1: # FAISS returns -1 if it can't find enough neighbors
            results.append(GLOBAL_DOCS[idx])
            
    return results

# Run the pipeline
create_faiss_index(INDEX_PATH, DOCSTORE_PATH)

# Test the search!
print("\n--- SEARCH RESULTS ---")
matches = search_faiss("Where did Dunk and Egg compete?", k=1)
for i, match in enumerate(matches):
    print(f"Result {i+1}:\n{match}")