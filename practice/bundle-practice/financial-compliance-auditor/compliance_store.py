from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

class ComplianceVectorStore:
    def __init__(self, rules_path: str):
        self.rules_path = rules_path
        self.vector_store = None

    def initialize_store(self) -> None:
        """
        Load the markdown rules, chunk them (chunk_size=200, overlap=20), 
        and create the FAISS index. (Standard implementation).
        """
        if not Path(self.rules_path).is_file():
            raise FileNotFoundError("File not found!")
        loaded_file = Path(self.rules_path).read_text(encoding="utf-8")
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_text(loaded_file)
        docs = [Document(page_content=chunk) for chunk in chunks]
        embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(documents=docs, embeddings=embeddings)
        return

    async def get_policy(self, query: str) -> str:
        """
        Retrieve the single most relevant document (k=1) and return its page_content.
        """
        # Call FAISS asimilarity_search
        docs = await self.vector_store.asimilarity_search(query, k=1)
        
        # Return the string directly
        return docs[0].page_content if docs else ""