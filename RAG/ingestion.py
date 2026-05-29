import os 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "ashford_tourney.txt"
with open(file_path, "w") as f:
    f.write("""The Tourney at Ashford Meadow was a chivalric tournament held in the Reach.
It was famously attended by Ser Duncan the Tall, a hedge knight of dubious knighthood.
Dunk's squire was a bald boy named Egg, who was secretly Prince Aegon Targaryen.

During the tourney, Ser Duncan attacked Prince Aerion Targaryen to defend an innocent puppeteer named Tanselle.
Because striking a prince of the blood is treason, Dunk demanded a Trial by Combat.
This escalated into a Trial of Seven, where Dunk had to find six other knights to fight beside him.

Tragically, Prince Baelor Breakspear championed Dunk and died from a blow to the head delivered by his own brother, Prince Maekar.
Dunk survived, won his freedom, and continued wandering Westeros with Egg.""")
    
print("--- Example : Document loaders and splitters ---")
loader = TextLoader(file_path)
docs = loader.load()
print(f"Loaded {len(docs)} document(s).")
print(f"Total characters in document: {len(docs[0].page_content)}\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=30,
    length_function=len,
    is_separator_regex=False
)

chunks = text_splitter.split_documents(docs)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} (Length: {len(chunk.page_content)}) ---")
    print(chunk.page_content)
    print("--------------------------------------------------\n")