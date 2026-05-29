
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "ashford_tourney.txt"

loader = TextLoader(file_path)
docs = loader.load()

print(f"Loaded {len(docs)} document(s).")
print(f"Total characters in document: {len(docs[0].page_content)}\n")

splitters = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=30,
    separators=["."],
    keep_separator=True,
    length_function=len,
    is_separator_regex=False
)
chunks = splitters.split_documents(docs)

print(f"Number of chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} (Length: {len(chunk.page_content)}) ---")
    print(chunk.page_content)
    print("--------------------------------------------------\n")