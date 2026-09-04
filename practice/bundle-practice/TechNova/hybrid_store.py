from typing import List
from langchain_core.documents import Document
# Hint: You will likely need FAISS and OpenAIEmbeddings here
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class TechNovaVectorStore:
    def __init__(self):
        self.vector_store = None

    def create_store(self, documents: List[Document]) -> None:
        """
        Initialize FAISS with OpenAI embeddings and store the documents.

        Args:
            documents: List of chunked Document objects.
        """
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(documents, embeddings)
        # raise NotImplementedError

    async def advanced_search(self, query: str, doc_type: str = None, k: int = 4) -> List[Document]:
        """
        Retrieve documents using Max Marginal Relevance (MMR) and metadata filtering.

        Constraints:
        1. Use MMR search to ensure diverse results.
        2. If doc_type is provided, strictly filter the search to only return 
           documents where metadata["source"] == doc_type.

        Args:
            query: The user's search query.
            doc_type: Optional filter (e.g., "policy" or "catalog").
            k: Number of diverse documents to return.

        Returns:
            List[Document]: The top k diverse and relevant documents.
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Please create the store first.")

        search_filter = {"source" : doc_type} if doc_type else None
        results = await self.vector_store.max_marginal_relevance_search(query, k=k, filter=search_filter)
        return results