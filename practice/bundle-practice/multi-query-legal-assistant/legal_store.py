from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class LegalVectorStore:
    def __init__(self):
        self.vector_store = None

    def initialize_store(self, documents: List[Document]) -> None:
        """Initialize FAISS with embeddings."""
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2")
        self.vector_store = FAISS.from_documents(documents=documents, embeddings=embeddings)
        return

    async def fetch_docs(self, query: str, k: int = 2) -> List[Document]:
        """
        Core Requirement:
        Perform a standard async similarity search returning top k docs.
        """
        return await self.vector_store.asimilarity_search(query=query, k=k)