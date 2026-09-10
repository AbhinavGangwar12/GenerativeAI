from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class MedicalVectorStore:
    def __init__(self):
        self.vector_store = None

    def initialize_store(self, documents: List[Document]) -> None:
        """Initialize FAISS with HuggingFace embeddings."""
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(documents, embeddings)
        return

    async def search(self, query: str, k: int = 2) -> List[Document]:
        """Perform standard async FAISS similarity search."""
        return await self.vector_store.asimilarity_search(query, k=k)