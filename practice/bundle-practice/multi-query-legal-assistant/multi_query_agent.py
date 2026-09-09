from typing import List
from .legal_store import LegalVectorStore
# Hint: You will need ChatPromptTemplate, ChatOpenAI/Ollama, StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LegalDiscoveryAgent:
    def __init__(self, vector_store: LegalVectorStore):
        self.vector_store = vector_store
        # TODO: Initialize your LLM here
        self.llm = ChatOllama(model="lamma3.1")
        # raise NotImplementedError

    async def _generate_query_variations(self, original_query: str) -> List[str]:
        """
        ADDITIONAL CHALLENGE Step 1: Query Expansion.
        
        Pass the original_query to the LLM and ask it to generate exactly 3 
        alternative ways to phrase the question using formal legal terminology.
        
        Returns:
            List[str]: A list containing the original query PLUS the 3 variations 
            (4 queries total).
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system" , "You are a Legal Assistant. Generate exactly 3 alternative ways to phrase the user's question using formal legal terminology. Output ONLY the 3 questions separated by newlines, with no numbers, bullet points, or extra text."),
            ("human" , "{question}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({"question" : original_query})
        
        # Split the string by newlines into a real Python list
        variations = response.strip().split('\n')
        
        # Combine the original query with the variations
        queries = [original_query]
        queries.extend(variations)
        
        return queries

    async def _retrieve_and_deduplicate(self, queries: List[str]) -> List[Document]:
        """
        ADDITIONAL CHALLENGE Step 2: Parallel Retrieval & Deduplication.
        
        For every query in the list, fetch the top 2 documents from the vector store.
        Since similar queries will likely return the exact same chunks, you must 
        deduplicate the final pool of documents using the 'chunk_id' in their metadata.
        
        Returns:
            List[Document]: A flat, deduplicated list of unique Document chunks.
        """
        chunk_ids = set()
        retrieved_docs = []
        for query in queries:
            top_k = await self.vector_store.fetch_docs(query=query, k=2)
            for doc in top_k:
                if doc.metadata["chunk_id"] not in chunk_ids:
                    retrieved_docs.append(doc)
                    chunk_ids.add(doc.metadata["chunk_id"])
        return retrieved_docs

    async def investigate(self, user_query: str) -> str:
        """
        The Main Pipeline:
        1. Generate query variations.
        2. Retrieve and deduplicate documents across all variations.
        3. Combine the unique documents into a context string.
        4. Prompt the LLM to answer the user_query using the comprehensive context.
        
        Returns:
            str: The final legal analysis.
        """
        variations = await self._generate_query_variations(original_query=user_query)
        docs = await self._retrieve_and_deduplicate(queries=variations)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal assistant and your task is to investigate the user query based on the context provided from the legal documents, don't make your own answer if the retrieved documents are not supporting the user query then say 'I don't have enough information to answer your question.'"),
            ("human" , "<context>\n\n{context}\n\n</context>\n\nUser Query:\n{query}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return await chain.ainvoke({"context" : context, "query" : user_query})