from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HugginFaceEmbeddings

class ArchiveVectorStore:
    def __init__(self):
        self.vector_store = None

    def create_store(self, documents: List[Document]) -> None:
        """
        Initialize FAISS with embeddings and store the documents.

        Args:
            documents: List of chunked Document objects.
        """
        # raise NotImplementedError
        if self.vector_store:
            return
        embeddings = HugginFaceEmbeddings(model="sentence-transformer/all-MiniLm-L6-v2")
        self.vector_store = FAISS.from_documents(documents, embeddings)
        return

    async def advanced_search(self, query: str, doc_type: str = None, k: int = 3) -> List[Document]:
        """
        Retrieve documents using Max Marginal Relevance (MMR) and metadata filtering.

        Constraints:
        1. Use async MMR search to ensure diverse results.
        2. If doc_type is provided, strictly filter the search to only return 
           documents where metadata["source"] == doc_type.

        Args:
            query: The user's search query.
            doc_type: Optional filter (e.g., "lore" or "rule").
            k: Number of diverse documents to return.

        Returns:
            List[Document]: The top k diverse and relevant documents.
        """
        # raise NotImplementedError
        if not self.vector_store:
            raise ValueError("Vector store not found!")

        filer = {"source" : doc_type} if doc_type else None
        results = await self.vector_store.amax_marginal_relevance_search(query, k=k, filters=filer)
        return results