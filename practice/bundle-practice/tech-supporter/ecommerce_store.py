from typing import List, Dict
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class ParentChildVectorStore:
    def __init__(self):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def initialize_store(self, child_documents: List[Document]) -> None:
        """Create the FAISS index using ONLY the child documents."""
        if not self.vector_store:
            self.vector_store = FAISS.from_documents(child_documents, self.embeddings)
        return

    def save_local(self, folder_path: str) -> None:
        """
        PERSISTENCE CHALLENGE: Save the FAISS vector store locally.
        Raise ValueError if the store hasn't been initialized.
        """
        if not Path(folder_path).is_dir():
            self.vector_store.save_local(folder_path)

    def load_local(self, folder_path: str) -> None:
        """
        PERSISTENCE CHALLENGE: Load the FAISS vector store from the local folder.
        """
        if Path(folder_path).is_dir():
            self.vector_store = FAISS.load_local(folder_path, self.embeddings, allow_dangerous_deserialization=True)

    async def retrieve_parents(self, query: str, parent_db: Dict[str, str], k: int = 2) -> List[str]:
        """
        SMALL-TO-BIG RETRIEVAL CHALLENGE:
        1. Search FAISS for the top `k` most relevant CHILD chunks.
        2. Extract the `parent_id` from the metadata of each retrieved chunk.
        3. Look up the full parent text in the `parent_db` using the extracted IDs.
        4. Deduplicate the parent texts (in case multiple child chunks belong to the same parent).
        
        Returns:
            List[str]: A list of the FULL, comprehensive parent documents.
        """
        docs = await self.vector_store.asimilarity_search(query=query, k=k)
        parent_ids = []
        for doc in docs:
            id = doc.meatadata.get("parent_id")
            if id not in parent_ids:
                parent_ids.append(id)
        return [parent_db[p_id] for p_id in parent_ids] 