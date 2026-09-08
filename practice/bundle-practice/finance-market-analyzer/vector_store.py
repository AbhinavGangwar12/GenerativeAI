from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class FinanceVectorStore:
    def __init__(self):
        self.vector_store = None

    def initialize_store(self, documents: List[Document]) -> None:
        """Initialize FAISS with HuggingFace embeddings."""
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2")
        self.vector_store = FAISS.from_documents(documents, embeddings)
        return 

    async def fetch_pool(self, query: str, pool_size: int = 10) -> List[Document]:
        """
        Core Requirement:
        Perform a standard async similarity search to fetch a large pool of documents 
        (more than what the user ultimately wants) to be sorted later.
        """
        if not self.vector_store:
            raise ValueError("Vector Store not found")
        return await self.vector_store.asimilarity_search(query, k=pool_size)