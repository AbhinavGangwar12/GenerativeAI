from typing import List
from .vector_store import FinanceVectorStore
import heapq

class TimeWeightedRetriever:
    def __init__(self, vector_store: FinanceVectorStore):
        self.vector_store = vector_store

    async def retrieve_and_sort(self, query: str, k: int = 2) -> List[Document]:
        """
        ADDITIONAL CHALLENGE: Hybrid Retrieval.
        
        The relevance of a document depends on two factors:
        1. Semantic similarity.
        2. The recency of the document (fiscal_year).
        
        Steps:
        1. Fetch a larger pool of semantically relevant documents from the vector store 
           (e.g., fetch the top 10).
        2. Sort this pool of documents so that the MOST RECENT documents appear first. 
           (If years are tied, maintain their original semantic ranking order).
        3. Return exactly `k` documents. 
        
        Args:
            query: The search query.
            k: The exact number of documents to return to the user.
            
        Returns:
            List[Document]: Top k documents, semantically relevant, ordered by recency.
        """
        docs = self.vector_store.fetch_pool(query)
        docs.sort(key=lambda x : x.metadata['fiscal_year'], reversed=True)

        return docs[:k]