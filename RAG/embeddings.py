from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "ashford_tourney.txt"
loader = TextLoader(file_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=30,separators=["."],
    keep_separator=False,
    length_function=len,
    is_separator_regex=False
)
documents = splitter.split_documents(docs)

print("Loading the embedding model... ")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Creating a vector store and adding the documents...")
vector_store = Chroma.from_documents(collection_name="ashford_tourney", embedding=embeddings, documents=documents)
query = "Who was the young boy serving the big knight?"
print(f"Query: {query}\n")

results = vector_store.similarity_search(query, k=2)

for i, result in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(result.page_content)
    print("--------------------------------------------------\n")


'''
Modify the chunks creation step. Add metadata dictionaries to the Documents.
Give chunks 1 and 2 the metadata: {"chapter": 1, "POV": "Dunk"}.
Give chunks 3 and 4 the metadata: {"chapter": 2, "POV": "Egg"}.

Embed them into a new Chroma vector store.

Run a similarity_search. However, look up the Chroma documentation (or use your intuition) to pass a filter argument into the search function. Ask a question, but force the vector store to only return results where the "POV" is "Egg".
'''


#     # The manual assignment
# chunks[0].metadata = {"chapter": 1, "POV": "Dunk"}
# chunks[2].metadata = {"chapter": 2, "POV": "Egg"}

# # The Chroma Filter Syntax
# results = vectorstore.similarity_search(
#     "Who was at the tourney?", 
#     k=1, 
#     filter={"POV": "Egg"} # This forces Chroma to ignore Dunk's chunks entirely
# )