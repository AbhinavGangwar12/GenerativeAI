from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class RunbookVectorStore:
    def __init__(self):
        self.vector_store = None

    def create_store(self, documents: List[Document]) -> None:
        """Initialize FAISS with embeddings and store the documents."""
        if self.vector_store:
            return
        embeddings = HuggingFaceEmbeddings(model="sentence-transformer/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(documents, embeddings)
        return

    async def search_runbooks(self, query: str, k: int = 2) -> List[Document]:
        """
        Retrieve top k runbooks related to the specific error query.
        """
        if not self.vector_store:
            raise ValueError("Vector Base not found!")
        response = await self.vector_store.asmilarity_search(query, k=k)
        return response